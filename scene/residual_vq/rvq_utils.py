# rvq_io.py
# I/O utilities for Residual K-Means VQ artifacts.
# - Save/load codebooks (multi-layer, per-parameter)
# - Optional: save indices per layer for analysis
# - Manifest with hashes/dtypes/shapes for integrity checks

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import os
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import torch


__all__ = [
    "save_rvq_artifacts",
    "load_rvq_codebooks",
    "load_manifest",
    "validate_codebooks_against_manifest",
]


# ----------------------------
# small helpers
# ----------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _iso_now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tensor_sha256(t: torch.Tensor) -> str:
    """Hash over raw bytes for integrity. Device/dtype-agnostic (bytes reflect dtype)."""
    b = t.detach().cpu().numpy().tobytes()
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def _as_cpu_list_tensors(seq: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [x.detach().cpu() for x in seq]


# ----------------------------
# public API
# ----------------------------
def save_rvq_artifacts(
    out_dir: str,
    codebooks: Dict[str, List[torch.Tensor]],
    indices: Optional[Dict[str, List[torch.LongTensor]]] = None,
    manifest_meta: Optional[Dict] = None,
    *,
    version: str = "rvq/v1",
    distance: str = "L2",
    layout: str = "stacked_list",
    save_legacy_first_layer: bool = True,
) -> Dict[str, str]:
    """
    Save RVQ artifacts into out_dir/vq_results style folder.
    Files:
      - rvq_codebooks.pt         (required)
      - rvq_manifest.json        (required)
      - rvq_indices_{param}.npz  (optional, if indices provided)
      - kmeans_centers.pth       (optional legacy 1st-layer only)
      - kmeans_indices.npz       (optional legacy 1st-layer only)

    Args:
      codebooks: {"dc": [C0, C1, ...], "sh": [C0, C1, ...]}, each Cℓ: (Kℓ, D)
      indices:   {"dc": [idsL0, idsL1, ...], "sh": [...]}, each ids: (N,)
      manifest_meta: extra fields to merge into manifest (e.g., {"block_id": 7})
      version/distance/layout: metadata fields
      save_legacy_first_layer: also export 1st-layer centers/indices for older tooling

    Returns:
      dict of produced file paths.
    """
    _ensure_dir(out_dir)

    paths = {}
    # 1) Save codebooks (CPU tensors for portability)
    cb_cpu = {k: _as_cpu_list_tensors(v) for k, v in codebooks.items()}
    path_codebooks = os.path.join(out_dir, "rvq_codebooks.pt")
    torch.save(cb_cpu, path_codebooks)
    paths["codebooks"] = path_codebooks

    # 2) Save indices (optional, per-param NPZ with keys "L{i}")
    if indices is not None:
        for key, id_list in indices.items():
            if id_list is None or len(id_list) == 0:
                continue
            npz_payload = {f"L{i}": ids.detach().cpu().numpy() for i, ids in enumerate(id_list)}
            path_npz = os.path.join(out_dir, f"rvq_indices_{key}.npz")
            np.savez_compressed(path_npz, **npz_payload)
            paths[f"indices_{key}"] = path_npz

    # 3) Build and write manifest
    manifest = {
        "version": version,
        "created_at": _iso_now_utc(),
        "params": {},
    }
    if manifest_meta:
        # don't overwrite mandatory keys unintentionally
        for k, v in manifest_meta.items():
            if k not in manifest:
                manifest[k] = v

    for key, cbs in cb_cpu.items():
        if not cbs:
            continue
        D = int(cbs[0].shape[1])
        layers = [int(cb.shape[0]) for cb in cbs]
        hashes = [_tensor_sha256(cb) for cb in cbs]
        manifest["params"][key] = {
            "layers": layers,
            "D": D,
            "distance": distance,
            "dtype": _dtype_str(cbs[0]),
            "layout": layout,
            "hash": hashes,
        }

    path_manifest = os.path.join(out_dir, "rvq_manifest.json")
    with open(path_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    paths["manifest"] = path_manifest

    # 4) Optional legacy first-layer exports
    if save_legacy_first_layer:
        legacy_centers = {}
        for key, cbs in cb_cpu.items():
            if len(cbs) > 0:
                legacy_centers[key] = cbs[0]  # only first layer

        if len(legacy_centers) > 0:
            path_legacy_centers = os.path.join(out_dir, "kmeans_centers.pth")
            torch.save(legacy_centers, path_legacy_centers)
            paths["legacy_centers"] = path_legacy_centers

        if indices is not None:
            legacy_indices = {}
            for key, id_list in indices.items():
                if id_list and len(id_list) > 0:
                    legacy_indices[key] = id_list[0].detach().cpu().numpy()  # only first layer
            if len(legacy_indices) > 0:
                path_legacy_idx = os.path.join(out_dir, "kmeans_indices.npz")
                np.savez_compressed(path_legacy_idx, **legacy_indices)
                paths["legacy_indices"] = path_legacy_idx

    return paths


def load_rvq_codebooks(
    path_pt: str,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, List[torch.Tensor]]:
    """
    Load codebooks saved by save_rvq_artifacts(...).

    Args:
      path_pt: path to rvq_codebooks.pt
      device/dtype: optional cast/placement

    Returns:
      {"dc": [C0, C1, ...], "sh": [C0, C1, ...]}
    """
    cb = torch.load(path_pt, map_location="cpu")
    result: Dict[str, List[torch.Tensor]] = {}
    for k, lst in cb.items():
        tensors = []
        for t in lst:
            t2 = t
            if device is not None:
                t2 = t2.to(device)
            if dtype is not None:
                t2 = t2.to(dtype=dtype)
            tensors.append(t2)
        result[k] = tensors
    return result


def load_manifest(path_json: str) -> Dict:
    with open(path_json, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_codebooks_against_manifest(
    codebooks: Dict[str, List[torch.Tensor]],
    manifest: Dict,
    *,
    strict_dtype: bool = False,
) -> bool:
    """
    Cross-check codebooks vs. manifest:
      - param keys present
      - per-layer shapes (K,D) match
      - content hashes match
      - dtype optionally enforced

    Raises:
      ValueError on mismatch (with exact reason).

    Returns:
      True if all checks pass.
    """
    params = manifest.get("params", {})
    if not params:
        raise ValueError("Manifest missing 'params'")

    for key, cbs in codebooks.items():
        if key not in params:
            raise ValueError(f"Manifest missing param section: '{key}'")

        manifest_entry = params[key]
        man_layers: List[int] = list(manifest_entry.get("layers", []))
        man_D: int = int(manifest_entry.get("D", -1))
        man_dtype: str = str(manifest_entry.get("dtype", "")).lower()
        man_hashes: List[str] = list(manifest_entry.get("hash", []))

        if len(cbs) != len(man_layers):
            raise ValueError(f"[{key}] num layers mismatch: got {len(cbs)} vs manifest {len(man_layers)}")

        for i, C in enumerate(cbs):
            if C.ndim != 2:
                raise ValueError(f"[{key}] codebook L{i} must be 2D (K,D), got {tuple(C.shape)}")
            Ki, Di = int(C.shape[0]), int(C.shape[1])
            if Di != man_D:
                raise ValueError(f"[{key}] D mismatch at L{i}: got {Di} vs manifest {man_D}")
            if Ki != int(man_layers[i]):
                raise ValueError(f"[{key}] K mismatch at L{i}: got {Ki} vs manifest {man_layers[i]}")

            # dtype check (optional)
            if strict_dtype:
                got_dtype = _dtype_str(C).lower()
                if got_dtype != man_dtype:
                    raise ValueError(f"[{key}] dtype mismatch at L{i}: got {got_dtype} vs manifest {man_dtype}")

            # hash check (always)
            h = _tensor_sha256(C)
            if i >= len(man_hashes):
                raise ValueError(f"[{key}] manifest missing hash for L{i}")
            if h != man_hashes[i]:
                raise ValueError(f"[{key}] hash mismatch at L{i}: got {h} vs manifest {man_hashes[i]}")

    return True