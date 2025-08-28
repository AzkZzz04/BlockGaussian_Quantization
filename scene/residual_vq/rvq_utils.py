# rvq_io.py
# I/O utilities for Residual K-Means VQ artifacts.
# - Save/load codebooks (multi-layer, per-parameter)
# - Optional: save indices per layer for analysis
# - Manifest with hashes/dtypes/shapes for integrity checks

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Any

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
    Drop-in replacement. Adds:
      - strict validation (dims/dtypes/layer counts)
      - atomic writes (temp file + replace)
      - index dtype downcast (uint8/uint16/uint32)
      - richer manifest (K, N per layer)
      - contiguous CPU hashing
    """

    def _atomic_write_bytes(path: str, data: bytes):
        d = os.path.dirname(path)
        os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _atomic_save_json(path: str, obj: Dict):
        payload = json.dumps(obj, indent=2).encode("utf-8")
        _atomic_write_bytes(path, payload)

    def _atomic_save_npz(path: str, arrays: Dict[str, np.ndarray]):
        # np.savez_compressed doesn't expose file descriptor sync; write to tmp bytes then replace
        import io, zipfile
        tmp_bytes = io.BytesIO()
        # Re-implement minimal NPZ with compression for atomicity
        with zipfile.ZipFile(tmp_bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for k, arr in arrays.items():
                with io.BytesIO() as buf:
                    np.save(buf, arr, allow_pickle=False)
                    zf.writestr(k + ".npy", buf.getvalue())
        _atomic_write_bytes(path, tmp_bytes.getvalue())

    def _as_cpu_list_tensors_safe(lst: List[torch.Tensor]) -> List[torch.Tensor]:
        out = []
        for t in lst:
            if not isinstance(t, torch.Tensor):
                raise TypeError("codebook entry must be torch.Tensor")
            if t.numel() == 0:
                raise ValueError("empty codebook tensor")
            out.append(t.detach().contiguous().to("cpu"))
        return out

    def _downcast_index(ids: torch.LongTensor, K: int) -> np.ndarray:
        # Choose smallest safe dtype
        if K <= 256:
            return ids.detach().cpu().numpy().astype(np.uint8, copy=False)
        elif K <= 65536:
            return ids.detach().cpu().numpy().astype(np.uint16, copy=False)
        elif K <= 2**32:
            return ids.detach().cpu().numpy().astype(np.uint32, copy=False)
        else:
            return ids.detach().cpu().numpy().astype(np.uint64, copy=False)

    def _cb_shape_list(cbs: List[torch.Tensor]) -> List[Tuple[int,int]]:
        return [(int(cb.shape[0]), int(cb.shape[1])) for cb in cbs]

    def _validate(codebooks, indices):
        # Ensure per-param: same D and dtype across layers; indices layer counts match codebooks
        for p, cbs in codebooks.items():
            if not isinstance(cbs, (list, tuple)) or len(cbs) == 0:
                raise ValueError(f"{p}: empty codebook list")
            cbs_cpu = _as_cpu_list_tensors_safe(cbs)
            D0 = int(cbs_cpu[0].shape[1])
            dtype0 = cbs_cpu[0].dtype
            for i, cb in enumerate(cbs_cpu):
                if cb.dim() != 2:
                    raise ValueError(f"{p}: codebook L{i} must be 2D (K,D)")
                if int(cb.shape[1]) != D0:
                    raise ValueError(f"{p}: inconsistent D at L{i} ({int(cb.shape[1])} vs {D0})")
                if cb.dtype != dtype0:
                    raise ValueError(f"{p}: inconsistent dtype at L{i} ({cb.dtype} vs {dtype0})")
            if indices is not None and p in indices and indices[p] is not None:
                if len(indices[p]) not in (0, len(cbs_cpu)):
                    raise ValueError(f"{p}: indices layers ({len(indices[p])}) != codebooks ({len(cbs_cpu)})")

    # --- start ---
    os.makedirs(out_dir, exist_ok=True)
    _validate(codebooks, indices)

    paths: Dict[str, str] = {}

    # 1) Save codebooks (CPU, contiguous) atomically
    cb_cpu = {k: _as_cpu_list_tensors_safe(v) for k, v in codebooks.items()}
    path_codebooks = os.path.join(out_dir, "rvq_codebooks.pt")
    # Use torch.save to temp, then replace
    tmp_cb = path_codebooks + ".tmp"
    torch.save(cb_cpu, tmp_cb)
    os.replace(tmp_cb, path_codebooks)
    paths["codebooks"] = path_codebooks

    # 2) Save indices (per-param NPZ with keys "L{i}")
    if indices is not None:
        for key, id_list in indices.items():
            if not id_list:
                continue
            # Downcast per layer based on K
            if key not in cb_cpu:
                continue
            Ks = [int(cb.shape[0]) for cb in cb_cpu[key]]
            npz_payload = {}
            for i, ids in enumerate(id_list):
                if ids is None:
                    continue
                npz_payload[f"L{i}"] = _downcast_index(ids, Ks[i])
            if len(npz_payload) == 0:
                continue
            path_npz = os.path.join(out_dir, f"rvq_indices_{key}.npz")
            _atomic_save_npz(path_npz, npz_payload)
            paths[f"indices_{key}"] = path_npz

    # 3) Build manifest (with per-layer K, optional N if indices present)
    manifest: Dict[str, Any] = {
        "version": version,
        "created_at": _iso_now_utc(),
        "distance": distance,   # keep global for clarity
        "layout": layout,       # keep global for clarity
        "params": {},
    }
    if manifest_meta:
        for k, v in manifest_meta.items():
            if k not in manifest:
                manifest[k] = v

    for key, cbs in cb_cpu.items():
        D = int(cbs[0].shape[1])
        dtype_str = _dtype_str(cbs[0])
        Ks = [int(cb.shape[0]) for cb in cbs]
        hashes = [_tensor_sha256(cb.contiguous()) for cb in cbs]  # ensure contiguous
        Ns = None
        if indices is not None and key in indices and indices[key]:
            Ns = [int(ids.numel()) if ids is not None else 0 for ids in indices[key]]
        entry = {
            "layers": Ks,
            "D": D,
            "dtype": dtype_str,
            "hash": hashes,
        }
        if Ns is not None:
            entry["N"] = Ns
        manifest["params"][key] = entry

    path_manifest = os.path.join(out_dir, "rvq_manifest.json")
    _atomic_save_json(path_manifest, manifest)
    paths["manifest"] = path_manifest

    # 4) Legacy first-layer exports (only if L0 exists)
    if save_legacy_first_layer:
        legacy_centers = {}
        for key, cbs in cb_cpu.items():
            if len(cbs) > 0:
                legacy_centers[key] = cbs[0]
        if len(legacy_centers) > 0:
            path_legacy_centers = os.path.join(out_dir, "kmeans_centers.pth")
            tmp_legacy = path_legacy_centers + ".tmp"
            torch.save(legacy_centers, tmp_legacy)
            os.replace(tmp_legacy, path_legacy_centers)
            paths["legacy_centers"] = path_legacy_centers

        if indices is not None:
            legacy_indices = {}
            for key, id_list in indices.items():
                if not id_list:
                    continue
                ids0 = id_list[0]
                if ids0 is None:
                    continue
                K0 = int(cb_cpu[key][0].shape[0]) if key in cb_cpu and len(cb_cpu[key]) > 0 else None
                if K0 is None:
                    continue
                legacy_indices[key] = _downcast_index(ids0, K0)
            if len(legacy_indices) > 0:
                path_legacy_idx = os.path.join(out_dir, "kmeans_indices.npz")
                _atomic_save_npz(path_legacy_idx, legacy_indices)
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