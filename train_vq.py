import os
import json
import time
import shutil
import random
import argparse

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter 

from gaussian_renderer import render
from utils.image_utils import psnr
from utils.general_utils import get_expon_lr_func
from utils.loss_utils import l1_loss, ssim, src2ref
from scene.cameras import get_render_camera
from scene.gaussian_model import GaussianModel
from scene.scene_loader import SceneDataset, Scene
from scene.residual_vq import Quantize_RVQ
from utils.utils import parse_cfg, cal_local_cam_extent, save_cfg, read_pcdfile

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False


# -----------------------------
# Reproducibility
# -----------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def save_vq_results(vq_quantizers_dict: dict, out_dir: str):
    """Save RVQ/K-Means artifacts from a dict of per-param quantizers.

    Each quantizer is expected to implement at least:
      - get_codebooks() -> List[Tensor(K_l, D)]
      - get_all_indices() -> List[LongTensor(N)] or (layers, N)

    Args:
        vq_quantizers_dict: {'dc': Quantize_RVQ, 'sh': Quantize_RVQ, ...}
        out_dir: output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    try:
        # unified persistence function in your project
        from scene.residual_vq import save_rvq_artifacts
    except Exception as e:
        print(f"⚠️  save_rvq_artifacts not found: {e}")
        return

    codebooks_dict = {}
    indices_dict = {}
    layers_per_param = {}

    for param_name, quantizer in vq_quantizers_dict.items():
        try:
            cbs = quantizer.get_codebooks()
            ids = quantizer.get_all_indices()
        except AttributeError as e:
            print(f"⚠️  Quantizer for '{param_name}' lacks methods: {e}")
            continue
        if cbs is None or len(cbs) == 0:
            continue

        codebooks_dict[param_name] = cbs
        indices_dict[param_name] = ids
        layers_per_param[param_name] = len(cbs)

    if not codebooks_dict:
        print("⚠️  No VQ artifacts to save (empty dict).")
        return

    is_single_layer = all(n == 1 for n in layers_per_param.values())
    vq_type_name = "k_means" if is_single_layer else "residual_vq"

    manifest_meta = {
        'quantized_params': list(codebooks_dict.keys()),
        'vq_type': vq_type_name,
        'layers_per_param': layers_per_param
    }

    paths = save_rvq_artifacts(out_dir, codebooks_dict, indices_dict, manifest_meta)
    print(f"💾 VQ artifacts saved to {out_dir}")
    for key, path in paths.items():
        print(f"   • {key}: {os.path.basename(path)}")


@torch.no_grad()
def eval_metrics(local_gaussian,
                 train_views_info_list,
                 cfg,
                 bg,
                 device,
                 eval_views_info=None,
                 sample_k=20,
                 max_images=None,
                 use_eval_scale=True,
                 save_dir=None,
                 prefix="Final"):
    """
    Evaluate PSNR/SSIM/LPIPS (optional), automatically select evaluation images and print+save results.
    Selection logic:
      - If eval_views_info provided and non-empty → evaluate all
      - Otherwise → randomly sample sample_k from train_views_info_list
    """
    import random

    # 1) select evaluation images (views)
    if eval_views_info is not None and len(eval_views_info) > 0:
        selected_views = list(eval_views_info)
        select_src = f"validation ({len(selected_views)} views)"
    else:
        k = min(len(train_views_info_list), sample_k)
        selected_views = random.sample(train_views_info_list, k=k) if k > 0 else []
        select_src = f"random-train (k={k})"

    if max_images is None:
        max_images = len(selected_views)
    max_images = min(max_images, len(selected_views))

    # 2) build read-only DataLoader
    dataset = SceneDataset(
        selected_views,
        cfg.image_scale if use_eval_scale else 1.0,
        cfg.scene_scale
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         num_workers=cfg.num_workers, drop_last=False)

    # 3) LPIPS (optional)
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    except Exception:
        pass

    # 4) accumulate metrics per image
    psnr_sum, ssim_sum, lpips_sum, n = 0.0, 0.0, 0.0, 0
    for i, vinfo in enumerate(loader):
        if i >= max_images:
            break

        extrinsic = vinfo["extrinsic"].squeeze(0).to(device)
        intrinsic = vinfo["intrinsic"].squeeze(0).to(device)
        H = int(vinfo["image_height"].item())
        W = int(vinfo["image_width"].item())
        image_gt = vinfo["image"].squeeze(0).to(device)  # [3,H,W], 0~1

        cam = get_render_camera(H, W, extrinsic, intrinsic)
        pkg = render(cam, local_gaussian, cfg, bg)
        img_r = pkg["render"]  # [3,H,W], 0~1

        # PSNR
        psnr_val = psnr(img_r, image_gt).mean().item()
        psnr_sum += psnr_val

        # SSIM
        if FUSED_SSIM_AVAILABLE:
            ssim_val = fused_ssim(img_r.unsqueeze(0), image_gt.unsqueeze(0)).item()
        else:
            ssim_val = ssim(img_r, image_gt).item()
        ssim_sum += ssim_val

        # LPIPS (if available)
        if lpips_fn is not None:
            lp = lpips_fn(img_r.unsqueeze(0) * 2 - 1, image_gt.unsqueeze(0) * 2 - 1)
            lpips_sum += float(lp.mean().item())

        n += 1

    metrics = {
        "PSNR": (psnr_sum / n) if n > 0 else 0.0,
        "SSIM": (ssim_sum / n) if n > 0 else 0.0,
        "LPIPS": (lpips_sum / n) if (n > 0 and lpips_fn is not None) else None,
        "N": n,
        "source": select_src
    }

    # print results
    print(f"\n=== {prefix} Metrics ===")
    print(f"Source: {metrics['source']}")
    print(f"Images evaluated: {metrics['N']}")
    print(f"PSNR : {metrics['PSNR']:.4f}")
    print(f"SSIM : {metrics['SSIM']:.4f}")
    if metrics['LPIPS'] is not None:
        print(f"LPIPS: {metrics['LPIPS']:.4f}")
    else:
        print("LPIPS: (skipped, install `lpips` to enable)")

    # save JSON (optional)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{prefix.lower()}_metrics.json")
        try:
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"[Metrics] Saved to {out_path}")
        except Exception as e:
            print(f"[Warn] Failed to save metrics: {e}")

    return metrics


def reconstruct(cfg, block_id, block_bbx_expand, views_info_list, init_pcd, eval_views_info=None, device=torch.device("cuda")):
    block_bbx = block_bbx_expand
    tb_writer = None  # For TensorBoard, can use: SummaryWriter(cfg.output_dirpath)

    print(f"Reconstructing block {block_id}, Num block views: {len(views_info_list)}")
    point_cloud_path = os.path.join(cfg.output_dirpath, "point_cloud")

    # Local Gaussian
    local_gaussian = GaussianModel(sh_degree=cfg.sh_degree)
    bg_color = [1, 1, 1] if cfg.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    bg = torch.rand((3), device=device) if cfg.random_background else background

    save_cfg(cfg, block_id)

    # Datasets
    scene_dataset = SceneDataset(views_info_list, cfg.image_scale, cfg.scene_scale, cfg.iterations * cfg.batch_size, preload=cfg.preload)
    scene_dataloader = torch.utils.data.DataLoader(scene_dataset, batch_size=cfg.batch_size, shuffle=True,
                                                   num_workers=cfg.num_workers, drop_last=False, pin_memory=True)
    if eval_views_info is not None:
        eval_dataset = SceneDataset(eval_views_info, cfg.image_scale, cfg.scene_scale)
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False,
                                                      num_workers=cfg.num_workers, drop_last=False)

    # Block hyper params - only set if not already defined in config
    if not hasattr(cfg, 'position_lr_max_steps') or cfg.position_lr_max_steps is None:
        cfg.position_lr_max_steps = cfg.iterations
    if not hasattr(cfg, 'densify_until_iter') or cfg.densify_until_iter is None:
        cfg.densify_until_iter = cfg.iterations // 2
    if not hasattr(cfg, 'opacity_reset_interval') or cfg.opacity_reset_interval is None:
        cfg.opacity_reset_interval = max(cfg.iterations // 10, 3000)
    if not hasattr(cfg, 'densify_from_iter') or cfg.densify_from_iter is None:
        cfg.densify_from_iter = 1000
    if not hasattr(cfg, 'densification_interval') or cfg.densification_interval is None:
        cfg.densification_interval = 500
    if not hasattr(cfg, 'densify_grad_threshold') or cfg.densify_grad_threshold is None:
        cfg.densify_grad_threshold = 0.0002
    if not hasattr(cfg, 'min_opacity') or cfg.min_opacity is None:
        cfg.min_opacity = 0.005
    if not hasattr(cfg, 'densify_only_in_block') or cfg.densify_only_in_block is None:
        cfg.densify_only_in_block = True

    # Learning rate params - set defaults if None (like train.py)
    if not hasattr(cfg, 'feature_lr') or cfg.feature_lr is None:
        cfg.feature_lr = 0.0025
    if not hasattr(cfg, 'opacity_lr') or cfg.opacity_lr is None:
        cfg.opacity_lr = 0.025
    if not hasattr(cfg, 'scaling_lr') or cfg.scaling_lr is None:
        cfg.scaling_lr = 0.005
    if not hasattr(cfg, 'rotation_lr') or cfg.rotation_lr is None:
        cfg.rotation_lr = 0.001

    # Initialize Gaussian
    scene_extent = cal_local_cam_extent(views_info_list)
    print("Scene extent:", scene_extent)
    local_gaussian.create_from_pcd(init_pcd, scene_extent)
    local_gaussian.training_setup(cfg)

    # Loss schedulers
    depth_l1_weight = get_expon_lr_func(cfg.depth_l1_weight_init, cfg.depth_l1_weight_final, max_steps=cfg.iterations)
    reproj_l1_weight = get_expon_lr_func(cfg.reproj_l1_weight_init, cfg.reproj_l1_weight_final, max_steps=cfg.iterations)

    # VQ config - RVQ enabled by default
    def get_config(key, default):
        return getattr(cfg, key, default)

    quantized_params = get_config('quant_params', ['sh', 'dc'])
    rvq_layers = get_config('rvq_layers', 2)  # default uses RVQ
    n_cls_sh = get_config('kmeans_ncls_sh', 4096)
    n_cls_dc = get_config('kmeans_ncls_dc', 4096)
    n_it = get_config('kmeans_iters', 10)
    kmeans_st_iter = get_config('kmeans_st_iter', 1000)
    freq_cls_assn = get_config('kmeans_freq', 500)

    # IndexPrior configuration
    use_index_prior = get_config('use_index_prior', False)
    index_prior_config = {
        'd_model': get_config('index_prior_d_model', 64),
        'nhead': get_config('index_prior_nhead', 8),
        'num_layers': get_config('index_prior_num_layers', 2),
        'use_positional_encoding': get_config('index_prior_use_pos_enc', True)
    }
    
    # IndexPrior model loading
    index_priors = None
    if use_index_prior:
        prior_ckpt_dir = get_config('index_prior_ckpt_dir', None)
        if prior_ckpt_dir and os.path.exists(prior_ckpt_dir):
            try:
                from scripts.load_index_prior import load_priors
                vocab_size = max(n_cls_sh, n_cls_dc)  # use maximum vocabulary size
                index_priors = load_priors(
                    prior_ckpt_dir=prior_ckpt_dir,
                    layers=rvq_layers,
                    vocab=vocab_size,
                    device='cuda'
                )
                print(f"✅ Loaded IndexPrior models from {prior_ckpt_dir}")
            except Exception as e:
                print(f"⚠️  Failed to load IndexPrior models: {e}")
                print("   Continuing without IndexPrior...")
                index_priors = None
        else:
            print("⚠️  IndexPrior enabled but no checkpoint directory provided")
            print("   Continuing without pre-trained IndexPrior...")

    print("\n" + "=" * 60)
    print(f"🚀 RVQ CONFIGURATION FOR BLOCK {block_id}")
    print("=" * 60)
    print("📋 RVQ Parameters:")
    vq_type_name = "K-Means" if rvq_layers == 1 else "Residual VQ"
    print(f"   • Type: {vq_type_name}")
    print(f"   • RVQ Layers: {rvq_layers}")
    print(f"   • Quantized Parameters: {quantized_params}")
    print(f"   • SH Clusters: {n_cls_sh}")
    print(f"   • DC Clusters: {n_cls_dc}")
    print(f"   • k-means Iterations: {n_it}")
    print(f"   • k-means Start Iteration: {kmeans_st_iter}")
    print(f"   • Cluster Assignment Frequency: {freq_cls_assn}")
    print(f"   • K-Means Initialization: Full data (no sampling)")
    if use_index_prior:
        print("🔮 IndexPrior Configuration:")
        print(f"   • Enabled: {use_index_prior}")
        print(f"   • Model Dimension: {index_prior_config['d_model']}")
        print(f"   • Attention Heads: {index_prior_config['nhead']}")
        print(f"   • Transformer Layers: {index_prior_config['num_layers']}")
        print(f"   • Positional Encoding: {index_prior_config['use_positional_encoding']}")
    print("=" * 60 + "\n")

    # RVQ quantizers
    kmeans_quantizers = {}

    # get common parameters
    sh_band_weighting = get_config('sh_band_weighting', True)
    band_weight_alpha = get_config('band_weight_alpha', 0.15)
    layer_aware_training = get_config('layer_aware_training', True)

    if 'dc' in quantized_params:
        kmeans_quantizers['dc'] = Quantize_RVQ(
            which='dc',
            target_k=n_cls_dc,
            layers=rvq_layers,
            num_iters=n_it,
            sh_band_weighting=False,  # DC doesn't use band weighting
            band_weight_alpha=band_weight_alpha,
            layer_aware_training=layer_aware_training,
            use_index_prior=use_index_prior,
            index_prior_config=index_prior_config
        )
        # inject pre-trained IndexPrior models
        if index_priors is not None:
            kmeans_quantizers['dc']._rvq.set_index_priors(index_priors)

    if 'sh' in quantized_params:
        kmeans_quantizers['sh'] = Quantize_RVQ(
            which='sh',
            target_k=n_cls_sh,
            layers=rvq_layers,
            num_iters=n_it,
            sh_band_weighting=sh_band_weighting,  # SH uses band weighting
            band_weight_alpha=band_weight_alpha,
            layer_aware_training=layer_aware_training,
            use_index_prior=use_index_prior,
            index_prior_config=index_prior_config
        )
        # inject pre-trained IndexPrior models
        if index_priors is not None:
            kmeans_quantizers['sh']._rvq.set_index_priors(index_priors)

    # pre-compute reassignment iterations to avoid boolean checks every iteration
    reassign_iterations = set()
    # first assignment
    reassign_iterations.add(kmeans_st_iter + 1)
    # periodic reassignment
    for i in range(kmeans_st_iter + 1, cfg.iterations + 1, freq_cls_assn):
        reassign_iterations.add(i)

    # pre-compute layer-aware training enable iterations
    layer_aware_iterations = set()
    if layer_aware_training:
        for i in reassign_iterations:
            if i > kmeans_st_iter + 1000:  # enable condition: assign and past warm-up
                layer_aware_iterations.add(i)

    # Timer
    start_time = time.time()
    end_time = start_time  # fallback

    # Training loop
    for iter_idx, view_info in enumerate(tqdm(scene_dataloader, desc=f"Reconstructing:{block_id}")):
        iteration = iter_idx + 1

        # ---- LR & SH degree ----
        local_gaussian.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            local_gaussian.oneupSHdegree()

        # -------- RVQ online --------
        # Pre-RVQ baseline evaluation (important baseline)
        if iteration == kmeans_st_iter:
            print(f"\n📊 PRE-RVQ BASELINE EVALUATION (Iteration {iteration})")
            print("=" * 60)
            eval_metrics(
                local_gaussian,
                train_views_info_list=views_info_list,
                cfg=cfg,
                bg=bg,
                device=device,
                eval_views_info=eval_views_info,
                sample_k=50,
                max_images=None,
                save_dir=os.path.join(point_cloud_path, str(block_id)),
                prefix=f"Block_{block_id}_PreRVQ_Baseline"
            )
            print("=" * 60)

        # RVQ logic: online mode quantizes during training; export mode skips online quantization
        if cfg.vq_mode == 'online' and iteration >= kmeans_st_iter:
            assign = iteration in reassign_iterations
            layer_aware_enabled = iteration in layer_aware_iterations

            # Simplified quantizer management
            for param_name, quantizer in kmeans_quantizers.items():
                if layer_aware_enabled:
                    quantizer.enable_layer_aware_training(True)

                if param_name == 'dc':
                    quantizer.forward_dc(local_gaussian, assign=assign)
                elif param_name == 'sh':
                    quantizer.forward_frest(local_gaussian, assign=assign)

                if layer_aware_enabled:
                    quantizer.enable_layer_aware_training(False)

            # Disable mid-training evaluation (only keep pre-RVQ and final evaluation)

        batch_sample_num = view_info["extrinsic"].shape[0]

        # -------- Forward & Loss over batch --------
        local_gaussian.optimizer.zero_grad(set_to_none=True)
        densify_cache = []
        finite_batch = True
        loss_total_val = 0.0

        # Read pseudo configuration (compatible with old spelling), default off
        def _get_cfg(cfg_obj, key: str, default=None):
            return getattr(cfg_obj, key, default)

        use_pseudo = bool(_get_cfg(cfg, 'pseudo_loss', _get_cfg(cfg, 'pesudo_loss', False)))
        pseudo_start = int(_get_cfg(cfg, 'pseudo_loss_start', _get_cfg(cfg, 'pesudo_loss_start', 10_000)))

        for sample_idx in range(batch_sample_num):
            extrinsic = view_info["extrinsic"][sample_idx].to(device)
            intrinsic = view_info["intrinsic"][sample_idx].to(device)
            image_height = view_info["image_height"][sample_idx].item()
            image_width = view_info["image_width"][sample_idx].item()
            image_gt = view_info["image"][sample_idx].to(device)

            camera_render = get_render_camera(image_height, image_width, extrinsic, intrinsic)
            render_pkg = render(camera_render, local_gaussian, cfg, bg)
            image_rendered = render_pkg["render"]  # [3, H, W]

            # photometric loss
            l1_loss_photo = l1_loss(image_rendered, image_gt)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image_rendered.unsqueeze(0), image_gt.unsqueeze(0))
            else:
                ssim_value = ssim(image_rendered, image_gt)
            loss_photo = (1.0 - cfg.lambda_dssim) * l1_loss_photo + cfg.lambda_dssim * (1.0 - ssim_value)

            # scaling regularization (original: volume penalty)
            sc = local_gaussian.get_scaling
            loss_scaling = (torch.clamp(sc, 0.001, 10.0) ** 2).mean()

            # basic loss
            loss = loss_photo + 0.01 * loss_scaling

            # depth inverse loss (original: directly use rendered "inverse depth")
            if cfg.depth_inv_loss and not isinstance(view_info["depth_inv"][sample_idx], str):
                depth_rendered_inv = render_pkg["depth"].squeeze(0)  # original: no clamp, use directly
                depth_gt_inv = view_info["depth_inv"][sample_idx].to(device)
                l1_loss_depth = torch.abs(depth_gt_inv - depth_rendered_inv).mean()
                loss = loss + depth_l1_weight(iteration) * l1_loss_depth

            # ---------- Reprojection loss（stable, 4 dirs: ±x/±y） ----------
            if use_pseudo and iteration > pseudo_start:
                try:
                    # 1) First clamp depth then take inverse, avoid 0/extremely small values
                    depth_ref = torch.clamp(render_pkg["depth"].squeeze(0), 1e-6, 1e6)
                    depth_inv_ref = 1.0 / depth_ref

                    # 2) Pixel-level perturbation amplitude, adaptive and clamped (1px ~ 0.05*W)
                    fx = intrinsic[0, 0]
                    px_shift = 0.05 * image_width * torch.median(depth_inv_ref) / fx
                    px_shift = torch.clamp(
                        px_shift,
                        min=torch.tensor(1.0, device=device),
                        max=torch.tensor(image_width * 0.05, device=device)
                    )

                    zero = torch.tensor(0.0, device=device)
                    dirs = [
                        torch.stack((+px_shift, zero,      zero)),   # +x
                        torch.stack((-px_shift, zero,      zero)),   # -x
                        torch.stack((zero,      +px_shift, zero)),   # +y
                        torch.stack((zero,      -px_shift, zero)),   # -y
                    ]
                    dir_idx = random.randrange(4)
                    disturb = dirs[dir_idx]

                    # 3) Render perturbed viewpoint
                    cam_src = get_render_camera(image_height, image_width, extrinsic, intrinsic, disturb=disturb)
                    pkg_src = render(cam_src, local_gaussian, cfg, bg)
                    img_src = torch.clamp(pkg_src["render"], 0.0, 1.0)
                    depth_src = torch.clamp(pkg_src["depth"].squeeze(0), 1e-6, 1e6)
                    depth_inv_src = 1.0 / depth_src

                    # 4) Reprojection + valid pixel mask
                    reproj_depth, reproj_img = src2ref(
                        camera_render.intrinsic, camera_render.extrinsic, depth_inv_ref,
                        cam_src.intrinsic,      cam_src.extrinsic,      depth_inv_src,
                        img_src
                    )
                    valid = (
                        torch.isfinite(reproj_depth) &
                        torch.isfinite(reproj_img) &
                        torch.isfinite(image_gt) &
                        (reproj_depth > 0)
                    )
                    valid_px = int(valid.sum().item())

                    # 5) Include in loss only if sufficient pixels
                    if valid_px >= 1024:
                        diff = torch.abs(reproj_img - image_gt)
                        loss_reproj_photo = diff[valid].mean()
                        if torch.isfinite(loss_reproj_photo):
                            loss = loss + reproj_l1_weight(iteration) * loss_reproj_photo
                        elif iteration % 200 == 0:
                            print("[Warn] Reproj: masked loss non-finite → 0")
                    elif iteration % 200 == 0:
                        print(f"[Warn] Reproj: insufficient overlap (valid_px={valid_px}) → skip")
                except Exception as e:
                    if iteration % 200 == 0:
                        print(f"[Warn] Reproj exception → skip ({e})")
            # ---------- Reprojection end ----------

            # Numerical stability (optional printing)
            if not torch.isfinite(loss):
                finite_batch = False
                print(f"[Warn] Non-finite loss at iter {iteration}: {loss.detach().item()}")
                print(f"[Warn] Components: photo={loss_photo.item():.6f}, scaling={loss_scaling.item():.6f}")
                break

            loss.backward()
            loss_total_val += float(loss.detach())

            # densify stats
            if iteration < cfg.densify_until_iter:
                local_gaussian.add_densification_stats(
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"]
                )
            # cache for max_radii2D
            densify_cache.append((render_pkg["visibility_filter"], render_pkg["radii"]))

        if not finite_batch:
            local_gaussian.optimizer.zero_grad(set_to_none=True)
            continue  # skip this batch's optimization and subsequent VQ

        # -------- Densify & Prune --------
        if iteration < cfg.densify_until_iter:
            with torch.no_grad():
                for visibility_filter, radii in densify_cache:
                    idx_vis = torch.where(visibility_filter)[0]
                    if idx_vis.numel() > 0 and idx_vis.max() < local_gaussian.max_radii2D.shape[0]:
                        local_gaussian.max_radii2D[idx_vis] = torch.max(
                            local_gaussian.max_radii2D[idx_vis], radii[idx_vis]
                        )
                if iteration > cfg.densify_from_iter and iteration % cfg.densification_interval == 0:
                    size_threshold = 500 if iteration > cfg.opacity_reset_interval else None
                    block_bbx_ = block_bbx if cfg.densify_only_in_block else None
                    local_gaussian.densify_and_prune(cfg.densify_grad_threshold, cfg.min_opacity,
                                                     scene_extent, size_threshold, block_bbx_)
                if iteration % cfg.opacity_reset_interval == 0 and iteration > cfg.densify_from_iter:
                    if local_gaussian.get_xyz.shape[0] > 0:
                        local_gaussian.reset_opacity()

        # -------- Optimizer step --------
        if local_gaussian.get_xyz.shape[0] > 0:
            local_gaussian.optimizer.step()
            local_gaussian.optimizer.zero_grad(set_to_none=True)
        else:
            print(f"[Warn] No points at iter {iteration}, skipping optimizer step")
            break

        # -------- Logging --------
        if tb_writer:
            tb_writer.add_scalar(f"block_{block_id}/loss", loss_total_val, iteration)
            tb_writer.add_scalar(f"block_{block_id}/Npts", local_gaussian.get_xyz.shape[0], iteration)

        # -------- Periodic PSNR & Intermediate Eval --------
        # Disable periodic evaluation (only keep two evaluations: pre-RVQ and final save)

        # -------- Save at end --------
        if iteration == cfg.iterations:
            eval_metrics(
                local_gaussian,
                train_views_info_list=views_info_list,
                cfg=cfg,
                bg=bg,
                device=device,
                eval_views_info=eval_views_info,  # can be None
                sample_k=100,
                max_images=None,
                save_dir=os.path.join(point_cloud_path, str(block_id)),
                prefix=f"Block_{block_id}"
            )
            end_time = time.time()

            os.makedirs(os.path.join(point_cloud_path, str(block_id)), exist_ok=True)

            # Save: quantized version + original version
            save_attributes = ['xyz', 'f_dc', 'f_rest', 'opacity', 'scale', 'rotation']

            if cfg.vq_mode == 'online':
                # Quantization/reassignment completed during training, save directly
                if len(kmeans_quantizers) > 0 and iteration >= kmeans_st_iter:
                    print("\n" + "=" * 60)
                    print(f"💾 SAVING FINAL RESULTS WITH RVQ FOR BLOCK {block_id}")
                    print("=" * 60)

                    local_gaussian.save_ply(
                        os.path.join(point_cloud_path, str(block_id),
                                     f"point_cloud_{iteration:03d}_quantized.ply"),
                        save_q=quantized_params,
                        save_attributes=save_attributes
                    )
                    local_gaussian.save_ply(
                        os.path.join(point_cloud_path, str(block_id),
                                     f"point_cloud_{iteration:03d}_original.ply"),
                        save_q=[],
                        save_attributes=save_attributes
                    )
                    save_vq_results(
                        kmeans_quantizers,
                        os.path.join(point_cloud_path, str(block_id), "vq_results")
                    )
                else:
                    local_gaussian.save_ply(
                        os.path.join(point_cloud_path, str(block_id), f"point_cloud_{iteration:03d}.ply")
                    )
            else:
                # export mode: no quantization during training, execute offline quantization once and export
                print("\n" + "=" * 60)
                print(f"💾 EXPORTING RVQ ARTIFACTS FOR BLOCK {block_id}")
                print("=" * 60)

                # Build offline quantizers (reuse training configuration parameters)
                kmeans_quantizers = {}
                if 'dc' in quantized_params:
                    kmeans_quantizers['dc'] = Quantize_RVQ(
                        which='dc',
                        target_k=n_cls_dc,
                        layers=rvq_layers,
                        num_iters=n_it,
                        sh_band_weighting=False,
                        band_weight_alpha=band_weight_alpha,
                        layer_aware_training=layer_aware_training,
                        use_index_prior=False,
                        index_prior_config=None
                    )
                if 'sh' in quantized_params:
                    kmeans_quantizers['sh'] = Quantize_RVQ(
                        which='sh',
                        target_k=n_cls_sh,
                        layers=rvq_layers,
                        num_iters=n_it,
                        sh_band_weighting=sh_band_weighting,
                        band_weight_alpha=band_weight_alpha,
                        layer_aware_training=layer_aware_training,
                        use_index_prior=False,
                        index_prior_config=None
                    )

                for pname, quantizer in kmeans_quantizers.items():
                    if pname == 'dc':
                        quantizer.forward_dc(local_gaussian, assign=True)
                    elif pname == 'sh':
                        quantizer.forward_frest(local_gaussian, assign=True)

                local_gaussian.save_ply(
                    os.path.join(point_cloud_path, str(block_id),
                                 f"point_cloud_{iteration:03d}_quantized.ply"),
                    save_q=quantized_params,
                    save_attributes=save_attributes
                )
                local_gaussian.save_ply(
                    os.path.join(point_cloud_path, str(block_id),
                                 f"point_cloud_{iteration:03d}_original.ply"),
                    save_q=[],
                    save_attributes=save_attributes
                )
                save_vq_results(
                    kmeans_quantizers,
                    os.path.join(point_cloud_path, str(block_id), "vq_results")
                )

    print(f"Block {block_id} optimize finished, total num pts: {local_gaussian.get_xyz.shape[0]}")
    elapsed_time = end_time - start_time
    with open(os.path.join(cfg.output_dirpath, "time_consumption.txt"), "a") as file:
        file.write(f"block_id: {block_id}, total num pts: {local_gaussian.get_xyz.shape[0]}, elapsed_time:{elapsed_time:.6f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Reconstruction Process of View-based Gaussian Splatting (with VQ).")
    parser.add_argument("--config", "-c", type=str, default="./configs/vq_example.yaml", help="config filepath")
    parser.add_argument("--scene_dirpath", "-s", type=str, default=None, help="scene data dirpath")
    parser.add_argument("--output_dirpath", "-o", type=str, default=None, help="optimized result output dirpath")
    parser.add_argument("--block_ids", "-b", nargs="+", type=int, default=None)

    # RVQ parameters
    parser.add_argument('--rvq_layers', type=int, default=2,
                        help='Number of RVQ layers (1 = K-Means, 2+ = RVQ). Default: 2')

    parser.add_argument('--kmeans_st_iter', type=int, default=1000,
                        help='Start k-Means based vector quantization from this iteration (online mode only)')
    parser.add_argument('--kmeans_ncls_sh', type=int, default=4096,
                        help='Number of clusters for SH')
    parser.add_argument('--kmeans_ncls_dc', type=int, default=4096,
                        help='Number of clusters for DC')
    parser.add_argument('--kmeans_iters', type=int, default=10,
                        help='Number of k-Means iterations')
    parser.add_argument('--kmeans_freq', type=int, default=500,
                        help='Frequency (iters) of reassignment in online mode')

    parser.add_argument("--quant_params", nargs="+", type=str, default=['sh', 'dc'],
                        help='Parameters to quantize: sh, dc')

    # VQ mode: online (quantize during training) or export (quantize once at the end)
    parser.add_argument('--vq_mode', type=str, choices=['online', 'export'], default='online',
                        help='VQ mode: online (default) quantizes during training; export quantizes once at the end')

    # IndexPrior parameters
    parser.add_argument('--use_index_prior', action='store_true',
                        help='Enable IndexPrior for context-aware quantization')

    parser.add_argument('--index_prior_d_model', type=int, default=64,
                        help='IndexPrior model dimension (default: 64)')
    parser.add_argument('--index_prior_nhead', type=int, default=8,
                        help='IndexPrior attention heads (default: 8)')
    parser.add_argument('--index_prior_num_layers', type=int, default=2,
                        help='IndexPrior transformer layers (default: 2)')

    # Positional encoding switch (mutually exclusive, enabled by default)
    pe_grp = parser.add_mutually_exclusive_group()
    pe_grp.add_argument('--index_prior_use_pos_enc', dest='index_prior_use_pos_enc',
                        action='store_true', help='Enable positional encoding in IndexPrior')
    pe_grp.add_argument('--no_index_prior_use_pos_enc', dest='index_prior_use_pos_enc',
                        action='store_false', help='Disable positional encoding in IndexPrior')
    parser.set_defaults(index_prior_use_pos_enc=True)
    
    # IndexPrior checkpoint directory
    parser.add_argument('--index_prior_ckpt_dir', type=str, default=None,
                        help='Directory containing pre-trained IndexPrior checkpoints')

    args = parser.parse_args()

    # Merge args into cfg - simplified configuration merging
    cfg = parse_cfg(args)

    # Use unified configuration merging function
    def set_config_if_missing(key, value):
        if not hasattr(cfg, key):
            setattr(cfg, key, value)

    set_config_if_missing('rvq_layers', args.rvq_layers)
    set_config_if_missing('quant_params', args.quant_params)
    set_config_if_missing('kmeans_st_iter', args.kmeans_st_iter)
    set_config_if_missing('kmeans_ncls_sh', args.kmeans_ncls_sh)
    set_config_if_missing('kmeans_ncls_dc', args.kmeans_ncls_dc)
    set_config_if_missing('kmeans_iters', args.kmeans_iters)
    set_config_if_missing('kmeans_freq', args.kmeans_freq)
    set_config_if_missing('vq_mode', args.vq_mode)

    # IndexPrior configuration
    set_config_if_missing('use_index_prior', args.use_index_prior)
    set_config_if_missing('index_prior_d_model', args.index_prior_d_model)
    set_config_if_missing('index_prior_nhead', args.index_prior_nhead)
    set_config_if_missing('index_prior_num_layers', args.index_prior_num_layers)
    set_config_if_missing('index_prior_use_pos_enc', args.index_prior_use_pos_enc)

    # Scene & output
    scene = Scene(cfg.scene_dirpath, evaluate=cfg.evaluate, scene_scale=cfg.scene_scale)
    os.makedirs(cfg.output_dirpath, exist_ok=True)
    shutil.copy(args.config, os.path.join(cfg.output_dirpath, "config.yaml"))

    # Block-by-block
    blocks_info_jsonpath = os.path.join(cfg.output_dirpath, "blocks_info.json")
    with open(blocks_info_jsonpath, "r") as json_file:
        blocks_info = json.load(json_file)

    num_blocks = blocks_info["num_blocks"]
    block_ids = args.block_ids if args.block_ids is not None else range(0, num_blocks)

    for block_id in block_ids:
        block_info = blocks_info[str(block_id)]
        pcd_filepath = block_info["block_pcd_filepath"]
        block_bbx_expand = np.array(block_info["bbx_expand"])
        pcd = read_pcdfile(pcd_filepath)
        views_info_list = [scene.views_info[view_id] for view_id in block_info["views_id"]]
        eval_views_info = None
        reconstruct(cfg, block_id, block_bbx_expand, views_info_list, pcd, eval_views_info)


if __name__ == "__main__":
    main()