#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


import os
import json
import torch
import numpy as np
from torch import nn

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.utils import load_gs_ply
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.general_utils import calculate_block_transform

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except Exception:
    SparseGaussianAdam = None


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self._features_dc_q = torch.empty(0)
        self._features_rest_q = torch.empty(0)
        self.dc_cluster_ids = None
        self.sh_cluster_ids = None

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.setup_functions()

    def set_quantized_dc(self, features_dc_q: torch.Tensor, cluster_ids: torch.Tensor):
        if self._features_dc is not None:
            assert features_dc_q.shape[:2] == self._features_dc.shape[:2], \
                f"DC q-shape mismatch: got {features_dc_q.shape}, expect (N,1,*) like {self._features_dc.shape}"
        self._features_dc_q = features_dc_q
        self.dc_cluster_ids = cluster_ids

    def set_quantized_sh(self, features_rest_q: torch.Tensor, cluster_ids: torch.Tensor):
        if self._features_rest is not None:
            assert features_rest_q.shape[:2] == self._features_rest.shape[:2], \
                f"SH q-shape mismatch: got {features_rest_q.shape}, expect (N,K,*) like {self._features_rest.shape}"
        self._features_rest_q = features_rest_q
        self.sh_cluster_ids = cluster_ids

    def clear_quantized(self):
        dev = (
            self._xyz.device if self._xyz.numel() > 0 else
            (self._features_dc.device if self._features_dc is not None and self._features_dc.numel() > 0 else torch.device("cpu"))
        )
        self._features_dc_q = torch.empty(0, device=dev)
        self._features_rest_q = torch.empty(0, device=dev)
        self.dc_cluster_ids = None
        self.sh_cluster_ids = None

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self.get_features_dc
        features_rest = self.get_features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        if self._features_dc_q.numel() > 0:
            return self._features_dc_q
        elif self._features_dc is not None:
            return self._features_dc
        else:
            raise RuntimeError("No features available")

    @property
    def get_features_rest(self):
        if self._features_rest_q.numel() > 0:
            return self._features_rest_q
        elif self._features_rest is not None:
            return self._features_rest
        else:
            raise RuntimeError("No features available")

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        assert torch.cuda.is_available(), "This pipeline currently requires CUDA."
        dev = torch.device("cuda")
        self.spatial_lr_scale = spatial_lr_scale

        pts_np = np.asarray(pcd.points)
        col_np = np.asarray(pcd.colors)

        fused_point_cloud = torch.tensor(pts_np, dtype=torch.float32, device=dev)
        fused_color = RGB2SH(torch.tensor(col_np, dtype=torch.float32, device=dev))

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float32, device=dev)
        features[:, :3, 0] = fused_color
        features[:, :, 1:] = 0.0

        print("Number of points at initialisation:", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 1e-7)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=dev)
        rots[:, 0] = 1.0

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32, device=dev))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=dev)

    def training_setup(self, cfg):
        dev = self._xyz.device if self._xyz.numel() > 0 else torch.device("cuda")
        self.percent_dense = cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=dev)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=dev)

        param_groups = [
            {'params': [self._xyz],          'lr': cfg.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc],  'lr': cfg.feature_lr,                                "name": "f_dc"},
            {'params': [self._features_rest],'lr': cfg.feature_lr / 20.0,                        "name": "f_rest"},
            {'params': [self._opacity],      'lr': cfg.opacity_lr,                               "name": "opacity"},
            {'params': [self._scaling],      'lr': cfg.scaling_lr,                               "name": "scaling"},
            {'params': [self._rotation],     'lr': cfg.rotation_lr,                              "name": "rotation"},
        ]

        if self.optimizer_type == "default" or SparseGaussianAdam is None:
            self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        else:
            self.optimizer = SparseGaussianAdam(param_groups, lr=0.0, eps=1e-15)

        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=cfg.position_lr_init * self.spatial_lr_scale,
            lr_final=cfg.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=cfg.position_lr_delay_mult,
            max_steps=cfg.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, save_att=None, save_q=None):
        if save_att is None:
            save_att = ['xyz', 'f_dc', 'f_rest', 'opacity', 'scale', 'rotation']
        if save_q is None:
            save_q = []

        q_dc = ('dc' in save_q) and (self._features_dc_q.numel() > 0) and (self.dc_cluster_ids is not None)
        q_sh = ('sh' in save_q) and (self._features_rest_q.numel() > 0) and (self.sh_cluster_ids is not None)

        l = []
        if 'xyz' in save_att:
            l += ['x', 'y', 'z']

        if 'f_dc' in save_att:
            if q_dc:
                l.append('f_dc_idx')
            else:
                for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
                    l.append(f'f_dc_{i}')

        if 'f_rest' in save_att:
            if q_sh:
                l.append('f_rest_idx')
            else:
                for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
                    l.append(f'f_rest_{i}')

        if 'opacity' in save_att:
            l.append('opacity')

        if 'scale' in save_att:
            for i in range(self._scaling.shape[1]):
                l.append(f'scale_{i}')

        if 'rotation' in save_att:
            for i in range(self._rotation.shape[1]):
                l.append(f'rot_{i}')

        return l

    def save_ply(self, path, save_q=None, save_attributes=None):
        if save_q is None:
            save_q = []
        os.makedirs(os.path.dirname(path), exist_ok=True)

        q_dc = ('dc' in save_q) and (self._features_dc_q.numel() > 0) and (self.dc_cluster_ids is not None)
        q_sh = ('sh' in save_q) and (self._features_rest_q.numel() > 0) and (self.sh_cluster_ids is not None)

        all_arrays = {}
        all_dtypes = {}

        xyz = self._xyz.detach().cpu().numpy()
        all_arrays['xyz'] = xyz
        all_dtypes['xyz'] = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

        if q_dc:
            arr = self.dc_cluster_ids.detach().cpu().numpy().astype(np.int32).reshape(-1, 1)
            all_arrays['f_dc'] = arr
            all_dtypes['f_dc'] = [('f_dc_idx', 'i4')]
        else:
            arr = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            all_arrays['f_dc'] = arr
            all_dtypes['f_dc'] = [('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')]

        if q_sh:
            arr = self.sh_cluster_ids.detach().cpu().numpy().astype(np.int32).reshape(-1, 1)
            all_arrays['f_rest'] = arr
            all_dtypes['f_rest'] = [('f_rest_idx', 'i4')]
        else:
            arr = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            ncols = arr.shape[1]
            all_arrays['f_rest'] = arr
            all_dtypes['f_rest'] = [(f'f_rest_{i}', 'f4') for i in range(ncols)]

        opacities = self._opacity.detach().cpu().numpy()
        all_arrays['opacity'] = opacities
        all_dtypes['opacity'] = [('opacity', 'f4')]

        scale = self._scaling.detach().cpu().numpy()
        all_arrays['scale'] = scale
        all_dtypes['scale'] = [(f'scale_{i}', 'f4') for i in range(scale.shape[1])]

        rotation = self._rotation.detach().cpu().numpy()
        all_arrays['rotation'] = rotation
        all_dtypes['rotation'] = [(f'rot_{i}', 'f4') for i in range(rotation.shape[1])]

        if save_attributes is None:
            save_attributes = ['xyz', 'f_dc', 'f_rest', 'opacity', 'scale', 'rotation']

        dtype_fields = []
        for key in save_attributes:
            dtype_fields.extend(all_dtypes[key])

        cols = [all_arrays[key] for key in save_attributes]
        attributes = np.concatenate(cols, axis=1)

        elements = np.empty(attributes.shape[0], dtype=dtype_fields)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        print('non-quantized attributes:', [k for k in save_attributes if k not in []])
        print('quantized attributes:', save_q)
        if q_dc:
            print('DC: saving indices → f_dc_idx (int32)')
        if q_sh:
            print('SH: saving indices → f_rest_idx (int32)')

    def load_ply(self, path, load_quant=False):
        plydata = PlyData.read(path)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        prop = plydata.elements[0]
        prop_names = [p.name for p in prop.properties]

        xyz = np.stack((np.asarray(prop["x"]), np.asarray(prop["y"]), np.asarray(prop["z"])), axis=1)
        opacities = np.asarray(prop["opacity"])[..., np.newaxis]

        has_dc_idx = "f_dc_idx" in prop_names
        has_sh_idx = "f_rest_idx" in prop_names

        if has_dc_idx:
            dc_idx = np.asarray(prop["f_dc_idx"]).astype(np.int64)
            features_dc = None
        else:
            f0 = np.asarray(prop["f_dc_0"])
            f1 = np.asarray(prop["f_dc_1"])
            f2 = np.asarray(prop["f_dc_2"])
            features_dc = np.stack([f0, f1, f2], axis=1)[:, :, None]
            dc_idx = None

        if has_sh_idx:
            sh_idx = np.asarray(prop["f_rest_idx"]).astype(np.int64)
            features_extra = None
        else:
            extra_f_names = sorted([n for n in prop_names if n.startswith("f_rest_")], key=lambda x: int(x.split('_')[-1]))
            F = len(extra_f_names)
            features_extra = np.zeros((xyz.shape[0], F), dtype=np.float32)
            for i, attr_name in enumerate(extra_f_names):
                features_extra[:, i] = np.asarray(prop[attr_name])
            features_extra = features_extra.reshape((features_extra.shape[0], 3, -1))
            sh_idx = None

        scale_names = sorted([n for n in prop_names if n.startswith("scale_")], key=lambda x: int(x.split('_')[-1]))
        rot_names = sorted([n for n in prop_names if n.startswith("rot_")], key=lambda x: int(x.split('_')[-1]))
        scales = np.stack([np.asarray(prop[n]) for n in scale_names], axis=1)
        rots = np.stack([np.asarray(prop[n]) for n in rot_names], axis=1)

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=dev).requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device=dev).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=dev).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float32, device=dev).requires_grad_(True))

        if features_dc is not None:
            self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float32, device=dev).transpose(1, 2).contiguous().requires_grad_(True))
            self._features_dc_q = torch.empty(0, device=dev)
            self.dc_cluster_ids = None
        else:
            self._features_dc = nn.Parameter(torch.zeros((xyz.shape[0], 1, 3), dtype=torch.float32, device=dev).requires_grad_(True))
            self._features_dc_q = torch.empty(0, device=dev)
            self.dc_cluster_ids = torch.from_numpy(dc_idx).to(device=dev, dtype=torch.long)

        if features_extra is not None:
            self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float32, device=dev).transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest_q = torch.empty(0, device=dev)
            self.sh_cluster_ids = None
        else:
            coeffs_minus1 = (self.max_sh_degree + 1) ** 2 - 1
            self._features_rest = nn.Parameter(torch.zeros((xyz.shape[0], coeffs_minus1, 3), dtype=torch.float32, device=dev).requires_grad_(True))
            self._features_rest_q = torch.empty(0, device=dev)
            self.sh_cluster_ids = torch.from_numpy(sh_idx).to(device=dev, dtype=torch.long)

        self.active_sh_degree = self.max_sh_degree

    def load_blocks_ply(self, plys_dirpath):
        xyz_all, features_dc_all, features_extra_all, opacities_all, scales_all, rots_all = [], [], [], [], [], []
        for block_plyfile in os.listdir(plys_dirpath):
            ply_filepath = os.path.join(plys_dirpath, block_plyfile)
            xyz, features_dc, features_extra, opacities, scales, rots = load_gs_ply(self.max_sh_degree, ply_filepath)
            xyz_all.append(xyz)
            features_dc_all.append(features_dc)
            features_extra_all.append(features_extra)
            opacities_all.append(opacities)
            scales_all.append(scales)
            rots_all.append(rots)

        xyz_all = np.concatenate(xyz_all, axis=0)
        features_dc_all = np.concatenate(features_dc_all, axis=0)
        features_extra_all = np.concatenate(features_extra_all, axis=0)
        opacities_all = np.concatenate(opacities_all, axis=0)
        scales_all = np.concatenate(scales_all, axis=0)
        rots_all = np.concatenate(rots_all, axis=0)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._xyz = torch.tensor(xyz_all, dtype=torch.float32, device=dev)
        self._features_dc = torch.tensor(features_dc_all, dtype=torch.float32, device=dev).transpose(1, 2).contiguous()
        self._features_rest = torch.tensor(features_extra_all, dtype=torch.float32, device=dev).transpose(1, 2).contiguous()
        self._opacity = torch.tensor(opacities_all, dtype=torch.float32, device=dev)
        self._scaling = torch.tensor(scales_all, dtype=torch.float32, device=dev)
        self._rotation = torch.tensor(rots_all, dtype=torch.float32, device=dev)
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] != name:
                continue
            old_param = group["params"][0]
            state = self.optimizer.state.get(old_param, {})
            exp_avg = state.get("exp_avg", torch.zeros_like(old_param))
            exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(old_param))
            exp_avg = torch.zeros_like(tensor) if exp_avg.shape != tensor.shape else exp_avg.to(tensor)
            exp_avg_sq = torch.zeros_like(tensor) if exp_avg_sq.shape != tensor.shape else exp_avg_sq.to(tensor)

            if old_param in self.optimizer.state:
                del self.optimizer.state[old_param]
            new_param = nn.Parameter(tensor.requires_grad_(True))
            group["params"][0] = new_param
            self.optimizer.state[new_param] = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
            optimizable_tensors[name] = new_param
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            old_param = group["params"][0]
            state = self.optimizer.state.get(old_param, None)
            if state is not None:
                state["exp_avg"] = state["exp_avg"][mask]
                state["exp_avg_sq"] = state["exp_avg_sq"][mask]
                del self.optimizer.state[old_param]
                new_param = nn.Parameter((old_param[mask].requires_grad_(True)))
                group["params"][0] = new_param
                self.optimizer.state[new_param] = state
                optimizable_tensors[group["name"]] = new_param
            else:
                group["params"][0] = nn.Parameter(old_param[mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optim = self._prune_optimizer(valid_points_mask)

        self._xyz = optim["xyz"]
        self._features_dc = optim["f_dc"]
        self._features_rest = optim["f_rest"]
        self._opacity = optim["opacity"]
        self._scaling = optim["scaling"]
        self._rotation = optim["rotation"]

        if self._features_dc_q.numel() > 0:
            self._features_dc_q = self._features_dc_q[valid_points_mask]
        if self._features_rest_q.numel() > 0:
            self._features_rest_q = self._features_rest_q[valid_points_mask]
        if self.dc_cluster_ids is not None:
            self.dc_cluster_ids = self.dc_cluster_ids[valid_points_mask]
        if self.sh_cluster_ids is not None:
            self.sh_cluster_ids = self.sh_cluster_ids[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            ext = tensors_dict[group["name"]]
            old_param = group["params"][0]
            state = self.optimizer.state.get(old_param, None)

            if state is not None:
                state["exp_avg"] = torch.cat((state["exp_avg"], torch.zeros_like(ext)), dim=0)
                state["exp_avg_sq"] = torch.cat((state["exp_avg_sq"], torch.zeros_like(ext)), dim=0)
                del self.optimizer.state[old_param]
                new_param = nn.Parameter(torch.cat((old_param, ext), dim=0).requires_grad_(True))
                group["params"][0] = new_param
                self.optimizer.state[new_param] = state
                optimizable_tensors[group["name"]] = new_param
            else:
                group["params"][0] = nn.Parameter(torch.cat((old_param, ext), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation
        }

        optim = self.cat_tensors_to_optimizer(d)
        self._xyz = optim["xyz"]
        self._features_dc = optim["f_dc"]
        self._features_rest = optim["f_rest"]
        self._opacity = optim["opacity"]
        self._scaling = optim["scaling"]
        self._rotation = optim["rotation"]

        if self._features_dc_q.numel() > 0:
            dev = self._features_dc.device
            new_dc_q = torch.zeros((new_features_dc.shape[0], self._features_dc_q.shape[1], self._features_dc_q.shape[2]), device=dev)
            self._features_dc_q = torch.cat((self._features_dc_q, new_dc_q), dim=0)
            new_dc_ids = torch.full((new_features_dc.shape[0],), -1, dtype=torch.long, device=dev)
            self.dc_cluster_ids = torch.cat((self.dc_cluster_ids, new_dc_ids), dim=0)
        if self._features_rest_q.numel() > 0:
            dev = self._features_rest.device
            new_rest_q = torch.zeros((new_features_rest.shape[0], self._features_rest_q.shape[1], self._features_rest_q.shape[2]), device=dev)
            self._features_rest_q = torch.cat((self._features_rest_q, new_rest_q), dim=0)
            new_rest_ids = torch.full((new_features_rest.shape[0],), -1, dtype=torch.long, device=dev)
            self.sh_cluster_ids = torch.cat((self.sh_cluster_ids, new_rest_ids), dim=0)

        dev = self._xyz.device
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=dev)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=dev)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=dev)

    def densify_and_split(self, grads, grad_threshold, scene_extent, block_bbx=None, N=2):
        n_init_points = self.get_xyz.shape[0]
        dev = self._xyz.device

        padded_grad = torch.zeros((n_init_points), device=dev)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = padded_grad >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        if block_bbx is not None:
            W2B, x_extent, y_extent, z_extent = calculate_block_transform(block_bbx)
            W2B = torch.tensor(W2B, device=dev, dtype=torch.float32)
            xyz_block = (W2B[:3, :3] @ self.get_xyz.T).T + W2B[:3, 3]
            block_bbx_mask = (
                (xyz_block[:, 0] >= -x_extent) & (xyz_block[:, 0] <= x_extent) &
                (xyz_block[:, 2] >= -z_extent) & (xyz_block[:, 2] <= z_extent) &
                (xyz_block[:, 1] >= -y_extent * 1.2) & (xyz_block[:, 1] <= y_extent * 1.2)
            )
            selected_pts_mask = torch.logical_and(block_bbx_mask, selected_pts_mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=dev)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self.get_features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self.get_features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=dev, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, block_bbx=None):
        dev = self._xyz.device

        selected_pts_mask = torch.norm(grads, dim=-1) >= grad_threshold
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        if block_bbx is not None:
            W2B, x_extent, y_extent, z_extent = calculate_block_transform(block_bbx)
            W2B = torch.tensor(W2B, device=dev, dtype=torch.float32)
            xyz_block = (W2B[:3, :3] @ self.get_xyz.T).T + W2B[:3, 3]
            block_bbx_mask = (
                (xyz_block[:, 0] >= -x_extent) & (xyz_block[:, 0] <= x_extent) &
                (xyz_block[:, 2] >= -z_extent) & (xyz_block[:, 2] <= z_extent) &
                (xyz_block[:, 1] >= -y_extent * 1.2) & (xyz_block[:, 1] <= y_extent * 1.2)
            )
            selected_pts_mask = torch.logical_and(block_bbx_mask, selected_pts_mask)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self.get_features_dc[selected_pts_mask]
        new_features_rest = self.get_features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, size_threshold, block_bbx=None):
        dev = self._xyz.device
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, block_bbx)
        self.densify_and_split(grads, max_grad, extent, block_bbx)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if size_threshold:
            big_points_vs = self.max_radii2D > size_threshold
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_ws), big_points_vs)

        self.prune_points(prune_mask)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if viewspace_point_tensor.grad is not None:
            self.xyz_gradient_accum[update_filter] += torch.norm(
                viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
            )
            self.denom[update_filter] += 1
        else:
            self.denom[update_filter] += 1

    def reset_opacity(self):
        if self._opacity.numel() > 0:
            dev = self._opacity.device
            opacities_new = self.inverse_opacity_activation(0.01 * torch.ones((self._opacity.shape[0], 1), dtype=torch.float32, device=dev))
            optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
            self._opacity = optimizable_tensors["opacity"]

    def clear_original_features(self):
        if self._features_dc_q.numel() > 0 and self._features_dc is not None:
            del self._features_dc
            self._features_dc = None
        if self._features_rest_q.numel() > 0 and self._features_rest is not None:
            del self._features_rest
            self._features_rest = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()