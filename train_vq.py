import os
import json
import time
import shutil
import random
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from gaussian_renderer import render
from utils.image_utils import psnr
from utils.general_utils import get_expon_lr_func
from utils.loss_utils import l1_loss, ssim, src2ref, loss_reproj
from scene.cameras import get_render_camera
from scene.gaussian_model import GaussianModel
from scene.scene_loader import SceneDataset, Scene
from scene.residual_vq import Quantize_RVQ
from utils.utils import parse_cfg, cal_local_cam_extent, save_cfg, read_pcdfile

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
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


def save_vq_results(kmeans_quantizers, quantized_params, out_dir):
    """Save RVQ results using unified approach."""
    os.makedirs(out_dir, exist_ok=True)

    # Use the RVQ I/O system (works for both single and multi-layer)
    from scene.residual_vq import save_rvq_artifacts
    
    codebooks_dict = {}
    indices_dict = {}
    
    for param in quantized_params:
        if param in kmeans_quantizers:
            q = kmeans_quantizers[param]
            if hasattr(q, 'get_codebooks') and hasattr(q, 'get_all_level_indices'):
                codebooks = q.get_codebooks()
                all_indices = q.get_all_level_indices()
                if codebooks and len(codebooks) > 0:
                    codebooks_dict[param] = codebooks
                if all_indices and len(all_indices) > 0:
                    indices_dict[param] = all_indices
    
    if codebooks_dict:
        # Determine if single layer (K-Means equivalent) or multi-layer RVQ
        is_single_layer = all(len(cb) == 1 for cb in codebooks_dict.values())
        vq_type_name = "K-Means" if is_single_layer else "Residual VQ"
        
        manifest_meta = {
            'quantized_params': quantized_params,
            'vq_type': vq_type_name.lower().replace('-', '_'),
            'layers_per_param': {p: len(codebooks_dict[p]) for p in codebooks_dict.keys()}
        }
        paths = save_rvq_artifacts(out_dir, codebooks_dict, indices_dict, manifest_meta)
        print(f"💾 {vq_type_name} results saved to {out_dir}")
        for key, path in paths.items():
            print(f"   • {key}: {os.path.basename(path)}")
    else:
        print(f"⚠️  No VQ codebooks to save")

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
    评估 PSNR / SSIM / LPIPS（可选），自动选择评估照片并打印+保存结果。

    选择逻辑：
      - 若提供 eval_views_info 且非空 → 全部评估
      - 否则 → 从 train_views_info_list 随机抽 sample_k 张（默认20）
    """
    import random

    # 1) 选择评估照片（视角）
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

    # 2) 构建只读 DataLoader
    dataset = SceneDataset(
        selected_views,
        cfg.image_scale if use_eval_scale else 1.0,
        cfg.scene_scale
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                         num_workers=cfg.num_workers, drop_last=False)

    # 3) LPIPS（可选）
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    except Exception:
        pass

    # 4) 逐张累积指标
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

        # LPIPS（若可用）
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

    # 打印结果
    print(f"\n=== {prefix} Metrics ===")
    print(f"Source: {metrics['source']}")
    print(f"Images evaluated: {metrics['N']}")
    print(f"PSNR : {metrics['PSNR']:.4f}")
    print(f"SSIM : {metrics['SSIM']:.4f}")
    if metrics['LPIPS'] is not None:
        print(f"LPIPS: {metrics['LPIPS']:.4f}")
    else:
        print("LPIPS: (skipped, install `lpips` to enable)")

    # 保存 JSON（可选）
    if save_dir is not None:
        import os, json
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
    tb_writer = None  # 想开 TensorBoard 可用：SummaryWriter(cfg.output_dirpath)

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

    # VQ config
    vq_enabled = hasattr(cfg, 'quant_params') or hasattr(cfg, 'kmeans_ncls_sh') or hasattr(cfg, 'kmeans_ncls_dc')
    
    if vq_enabled:
        quantized_params = cfg.quant_params if hasattr(cfg, 'quant_params') else ['sh', 'dc']
        rvq_layers = cfg.rvq_layers if hasattr(cfg, 'rvq_layers') else 2
        n_cls_sh = cfg.kmeans_ncls_sh if hasattr(cfg, 'kmeans_ncls_sh') else 4096
        n_cls_dc = cfg.kmeans_ncls_dc if hasattr(cfg, 'kmeans_ncls_dc') else 4096
        n_it = cfg.kmeans_iters if hasattr(cfg, 'kmeans_iters') else 10
        kmeans_st_iter = cfg.kmeans_st_iter if hasattr(cfg, 'kmeans_st_iter') else 1000
        freq_cls_assn = cfg.kmeans_freq if hasattr(cfg, 'kmeans_freq') else 500
        vq_mode = cfg.vq_mode if hasattr(cfg, 'vq_mode') else 'online'
        # max_samples parameter removed - now uses full data for optimal K-Means++ init
    else:
        quantized_params = []
        rvq_layers = 1
        n_cls_sh = 0
        n_cls_dc = 0
        n_it = 0
        kmeans_st_iter = float('inf')
        freq_cls_assn = float('inf')
        vq_mode = 'disabled'
        # max_samples parameter removed - K-Means uses full data

    if vq_enabled:
        print("\n" + "=" * 60)
        print(f"🚀 VQ CONFIGURATION FOR BLOCK {block_id}")
        print("=" * 60)
        print("📋 VQ Parameters:")
        vq_type_name = "K-Means" if rvq_layers == 1 else "Residual VQ"
        print(f"   • Type: {vq_type_name}")
        print(f"   • RVQ Layers: {rvq_layers}")
        print(f"   • Mode: {vq_mode}  (online: 训练中用量化渲染 / export: 仅导出前量化)")
        print(f"   • Quantized Parameters: {quantized_params}")
        print(f"   • SH Clusters: {n_cls_sh}")
        print(f"   • DC Clusters: {n_cls_dc}")
        print(f"   • k-means Iterations: {n_it}")
        print(f"   • k-means Start Iteration: {kmeans_st_iter}")
        print(f"   • Cluster Assignment Frequency: {freq_cls_assn}")
        print(f"   • K-Means Initialization: Full data (no sampling)")
        print("=" * 60 + "\n")
    else:
        print(f"\n[INFO] VQ disabled for block {block_id} - running in standard mode (like train.py)")

    # VQ quantizers (unified RVQ approach)
    kmeans_quantizers = {}
    if vq_enabled and 'dc' in quantized_params:
        # 获取层间感知和频带重加权参数
        sh_band_weighting = getattr(cfg, 'sh_band_weighting', True)
        band_weight_alpha = getattr(cfg, 'band_weight_alpha', 0.15)
        layer_aware_training = getattr(cfg, 'layer_aware_training', True)
        
        kmeans_quantizers['dc'] = Quantize_RVQ(
            which='dc', 
            target_k=n_cls_dc, 
            layers=rvq_layers, 
            num_iters=n_it,
            sh_band_weighting=False,  # DC不使用频带重加权
            band_weight_alpha=band_weight_alpha,
            layer_aware_training=layer_aware_training
        )
    
    if vq_enabled and 'sh' in quantized_params:
        # 获取层间感知和频带重加权参数
        sh_band_weighting = getattr(cfg, 'sh_band_weighting', True)
        band_weight_alpha = getattr(cfg, 'band_weight_alpha', 0.15)
        layer_aware_training = getattr(cfg, 'layer_aware_training', True)
        
        kmeans_quantizers['sh'] = Quantize_RVQ(
            which='sh', 
            target_k=n_cls_sh, 
            layers=rvq_layers, 
            num_iters=n_it,
            sh_band_weighting=sh_band_weighting,  # SH使用频带重加权
            band_weight_alpha=band_weight_alpha,
            layer_aware_training=layer_aware_training
        )

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

        # -------- VQ online --------
        # VQ启动前的预评估 (重要基线)
        if vq_enabled and vq_mode == 'online' and iteration == kmeans_st_iter:
            print(f"\n📊 PRE-VQ BASELINE EVALUATION (Iteration {iteration})")
            print("=" * 60)
            eval_metrics(
                local_gaussian,
                train_views_info_list=views_info_list,
                cfg=cfg,
                bg=bg,
                device=device,
                eval_views_info=eval_views_info,
                sample_k=50,  # 更详细的基线评估
                max_images=None,
                save_dir=cfg.output_dirpath,
                prefix=f"Block_{block_id}_PreVQ_Baseline"
            )
            print("=" * 60)

        if vq_enabled and vq_mode == 'online' and iteration >= kmeans_st_iter:
            assign = (iteration == kmeans_st_iter + 1) or (iteration % freq_cls_assn == 1)
            
            # Layer-aware training: enable during assignment iterations
            layer_aware_enabled = assign and iteration > kmeans_st_iter + 1000  # 启用条件：assign且过了warm-up
            
            if 'dc' in kmeans_quantizers:
                if layer_aware_enabled:
                    kmeans_quantizers['dc'].enable_layer_aware_training(True)
                kmeans_quantizers['dc'].forward_dc(local_gaussian, assign=assign)
                if layer_aware_enabled:
                    kmeans_quantizers['dc'].enable_layer_aware_training(False)
                    
            if 'sh' in kmeans_quantizers:
                if layer_aware_enabled:
                    kmeans_quantizers['sh'].enable_layer_aware_training(True)
                kmeans_quantizers['sh'].forward_frest(local_gaussian, assign=assign)
                if layer_aware_enabled:
                    kmeans_quantizers['sh'].enable_layer_aware_training(False)
            
            # VQ首次启动后的评估
            if assign and iteration == kmeans_st_iter + 1:
                print(f"\n🎯 POST-VQ INITIAL EVALUATION (Iteration {iteration})")
                print("=" * 60)
                eval_metrics(
                    local_gaussian,
                    train_views_info_list=views_info_list,
                    cfg=cfg,
                    bg=bg,
                    device=device,
                    eval_views_info=eval_views_info,
                    sample_k=30,
                    max_images=15,
                    save_dir=cfg.output_dirpath,
                    prefix=f"Block_{block_id}_PostVQ_Initial"
                )
                print("=" * 60)
            
            # 每1000轮打印层损失统计
            if layer_aware_enabled and iteration % 1000 == 0:
                for param_name, quantizer in kmeans_quantizers.items():
                    stats = quantizer.get_layer_loss_stats()
                    if stats:
                        print(f"[Layer-Aware {param_name.upper()}] Layer loss stats:")
                        for layer_name, layer_stat in stats.items():
                            print(f"  {layer_name}: mean={layer_stat['mean']:.4f}, count={layer_stat['count']}")
                            
                # 层间感知启用后的评估
                if iteration == kmeans_st_iter + 1000:
                    print(f"\n🧠 LAYER-AWARE TRAINING EVALUATION (Iteration {iteration})")
                    print("=" * 60)
                    eval_metrics(
                        local_gaussian,
                        train_views_info_list=views_info_list,
                        cfg=cfg,
                        bg=bg,
                        device=device,
                        eval_views_info=eval_views_info,
                        sample_k=30,
                        max_images=15,
                        save_dir=cfg.output_dirpath,
                        prefix=f"Block_{block_id}_LayerAware_Start"
                    )
                    print("=" * 60)

        # export mode hint
        if vq_enabled and vq_mode == 'export' and iteration % 200 == 0:
            remaining = max(kmeans_st_iter - iteration, 0)
            print(f"[VQ(export)] Will quantize only at export. Start threshold unused here. Remaining ~{remaining} iters.")

        batch_sample_num = view_info["extrinsic"].shape[0]

        # -------- Forward & Loss over batch --------
        local_gaussian.optimizer.zero_grad(set_to_none=True)
        densify_cache = []
        finite_batch = True
        loss_total_val = 0.0

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

            # scaling 正则（原版：体积惩罚）
            sc = local_gaussian.get_scaling
            loss_scaling = (torch.clamp(sc, 0.001, 10.0) ** 2).mean()

            # 基础损失
            loss = loss_photo + 0.01 * loss_scaling

            # depth inverse loss（原版直接用渲染的“倒深度”）
            if cfg.depth_inv_loss and not isinstance(view_info["depth_inv"][sample_idx], str):
                depth_rendered_inv = render_pkg["depth"].squeeze(0)  # 原版：不做 clamp 直接使用
                depth_gt_inv = view_info["depth_inv"][sample_idx].to(device)
                l1_loss_depth = torch.abs(depth_gt_inv - depth_rendered_inv).mean()
                loss = loss + depth_l1_weight(iteration) * l1_loss_depth

            
           # ---------- Reprojection loss（stable, 4 dirs: ±x/±y） ----------
            use_pseudo = cfg.pseudo_loss if hasattr(cfg, 'pseudo_loss') else cfg.pesudo_loss
            pseudo_start = cfg.pseudo_loss_start if hasattr(cfg, 'pseudo_loss_start') else cfg.pesudo_loss_start
            if use_pseudo and iteration > pseudo_start:
                try:
                    # 1) 先把深度夹紧再取倒数，避免 0/极小值
                    depth_ref = torch.clamp(render_pkg["depth"].squeeze(0), 1e-6, 1e6)
                    depth_inv_ref = 1.0 / depth_ref

                    # 2) 像素级扰动幅度，自适应并限幅（1px ~ 0.05*W）
                    fx = intrinsic[0, 0]
                    px_shift = 0.05 * image_width * torch.median(depth_inv_ref) / fx
                    px_shift = torch.clamp(
                        px_shift,
                        min=torch.tensor(1.0, device=device),
                        max=torch.tensor(image_width * 0.05, device=device)
                    )

                    zero = torch.tensor(0.0, device=device)
                    dirs = [
                        torch.stack((+px_shift, zero,      zero)),   # +x（像素右移）
                        torch.stack((-px_shift, zero,      zero)),   # -x（像素左移）
                        torch.stack((zero,      +px_shift, zero)),   # +y（像素下移）
                        torch.stack((zero,      -px_shift, zero)),   # -y（像素上移）
                    ]
                    # 用 CPU 取随机索引，避免张量->Python int 的设备问题
                    dir_idx = random.randrange(4)
                    disturb = dirs[dir_idx]

                    # 3) 渲染扰动视角，深度同样夹紧
                    cam_src = get_render_camera(image_height, image_width, extrinsic, intrinsic, disturb=disturb)
                    pkg_src = render(cam_src, local_gaussian, cfg, bg)
                    img_src = torch.clamp(pkg_src["render"], 0.0, 1.0)
                    depth_src = torch.clamp(pkg_src["depth"].squeeze(0), 1e-6, 1e6)
                    depth_inv_src = 1.0 / depth_src

                    # 4) 重投影 + 有效像素 mask
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

                    # 5) 只有足够像素才计入损失，并确保数值有限
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
            # ---------- 重投影结束 ----------

            # 数值稳定性（可选打印）
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
            continue  # 跳过这个 batch 的优化与 VQ 后续

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

        # -------- Periodic PSNR --------
        if iteration % 2000 == 0:
            psnr_sum, n = 0.0, 0
            for idx_eval, vinfo in enumerate(scene_dataloader):
                if idx_eval >= 2:
                    break
                bsz = vinfo["extrinsic"].shape[0]
                for s in range(bsz):
                    extrinsic = vinfo["extrinsic"][s].to(device)
                    intrinsic = vinfo["intrinsic"][s].to(device)
                    H = vinfo["image_height"][s].item()
                    W = vinfo["image_width"][s].item()
                    image_gt = vinfo["image"][s].to(device)
                    cam = get_render_camera(H, W, extrinsic, intrinsic)
                    pkg = render(cam, local_gaussian, cfg, bg)
                    img_r = pkg["render"]
                    psnr_sum += psnr(img_r, image_gt).mean().item()
                    n += 1
            if tb_writer and n > 0:
                tb_writer.add_scalar(f"block_{block_id}/train-PSNR", psnr_sum / n, iteration)
            
            # 完整的中间评估 (包含LPIPS)
            print(f"\n🔍 INTERMEDIATE EVALUATION AT ITERATION {iteration}")
            eval_metrics(
                local_gaussian,
                train_views_info_list=views_info_list,
                cfg=cfg,
                bg=bg,
                device=device,
                eval_views_info=eval_views_info,
                sample_k=20,  # 快速评估，少量图片
                max_images=10,
                save_dir=cfg.output_dirpath,
                prefix=f"Block_{block_id}_Iter_{iteration}"
            )

        # -------- Save at end --------
        if iteration == cfg.iterations:
            eval_metrics(
                local_gaussian,
                train_views_info_list=views_info_list,
                cfg=cfg,
                bg=bg,
                device=device,
                eval_views_info=eval_views_info,  # 可为 None
                sample_k=100,
                max_images=None,                  # None=选多少评多少
                save_dir=cfg.output_dirpath,
                prefix=f"Block_{block_id}"
            )
            end_time = time.time()

            os.makedirs(os.path.join(point_cloud_path, str(block_id)), exist_ok=True)

            # export-only 模式：仅在保存前量化并导出
            if vq_enabled and vq_mode == 'export' and len(kmeans_quantizers) > 0:
                print("\n" + "=" * 60)
                print(f"🎯 RUNNING K-MEANS (EXPORT) FOR BLOCK {block_id}")
                print("=" * 60)
                if 'dc' in kmeans_quantizers:
                    kmeans_quantizers['dc'].forward_dc(local_gaussian, assign=True)
                if 'sh' in kmeans_quantizers:
                    kmeans_quantizers['sh'].forward_frest(local_gaussian, assign=True)

            # 保存：量化版 + 原始版
            save_attributes = ['xyz', 'f_dc', 'f_rest', 'opacity', 'scale', 'rotation']

            if vq_enabled and len(kmeans_quantizers) > 0 and (vq_mode in ['online', 'export']) and iteration >= kmeans_st_iter:
                print("\n" + "=" * 60)
                print(f"💾 SAVING FINAL RESULTS WITH VQ FOR BLOCK {block_id}")
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
                save_vq_results(kmeans_quantizers, quantized_params,
                                os.path.join(point_cloud_path, str(block_id), "vq_results"))
            else:
                local_gaussian.save_ply(
                    os.path.join(point_cloud_path, str(block_id), f"point_cloud_{iteration:03d}.ply")
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

    # VQ parameters
    parser.add_argument('--vq_mode', type=str, default='online', choices=['online', 'export'],
                        help="VQ usage: 'online' uses quantized features during training; 'export' only quantizes at final export.")
    # Note: We now use unified RVQ (layers=1 equals K-Means)
    parser.add_argument('--rvq_layers', type=int, default=2,
                        help='Number of RVQ layers (1 = equivalent to K-Means, 2+ = true RVQ)')

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
    # kmeans_max_samples parameter removed - now always uses full data for optimal initialization

    parser.add_argument("--quant_params", nargs="+", type=str, default=['sh', 'dc'],
                        help='Parameters to quantize: sh, dc')

    args = parser.parse_args()

    # Merge args into cfg
    cfg = parse_cfg(args)
    if not hasattr(cfg, 'vq_mode'):
        cfg.vq_mode = args.vq_mode
    # Unified RVQ approach - remove vq_type
    if not hasattr(cfg, 'rvq_layers'):
        cfg.rvq_layers = args.rvq_layers
    if not hasattr(cfg, 'quant_params'):
        cfg.quant_params = args.quant_params
    if not hasattr(cfg, 'kmeans_st_iter'):
        cfg.kmeans_st_iter = args.kmeans_st_iter
    if not hasattr(cfg, 'kmeans_ncls_sh'):
        cfg.kmeans_ncls_sh = args.kmeans_ncls_sh
    if not hasattr(cfg, 'kmeans_ncls_dc'):
        cfg.kmeans_ncls_dc = args.kmeans_ncls_dc
    if not hasattr(cfg, 'kmeans_iters'):
        cfg.kmeans_iters = args.kmeans_iters
    if not hasattr(cfg, 'kmeans_freq'):
        cfg.kmeans_freq = args.kmeans_freq
    # kmeans_max_samples removed - full data initialization always used

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