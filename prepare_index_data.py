#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse, numpy as np, torch
from typing import Dict, List, Tuple, Optional

try:
    from utils.utils import read_pcdfile
except Exception:
    import sys
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT not in sys.path:
        sys.path.append(ROOT)
    from utils.utils import read_pcdfile


def find_block_dirs(root: str) -> List[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(fn.endswith(".ply") and "original" in fn for fn in filenames) \
           and "vq_results" in dirnames:
            out.append(dirpath)
    return sorted(out)


def load_indices(block_dir: str, which="dc", layer=0, dtype=np.int32) -> np.ndarray:
    p = os.path.join(block_dir, "vq_results", f"rvq_indices_{which}.npz")
    z = np.load(p)
    if f"L{layer}" in z.files:
        return z[f"L{layer}"].astype(dtype)

    L_keys = sorted(
        [k for k in z.files if k.startswith("L") and k[1:].isdigit()],
        key=lambda x: int(x[1:])
    )
    if L_keys:
        Lmax = len(L_keys)
        if not (0 <= layer < Lmax):
            raise IndexError(f"Requested layer={layer}, available 0..{Lmax-1} in {p}")
        stacked = np.stack([z[k] for k in L_keys], axis=1).astype(dtype)  # (N,L)
        return stacked[:, layer]

    key = "indices" if "indices" in z.files else ("data" if "data" in z.files else None)
    if key is None:
        raise KeyError(f"No valid index arrays found in {p}. Keys: {list(z.files)}")
    arr = z[key].astype(dtype)
    if arr.ndim == 1:
        arr = arr[:, None]
    if not (0 <= layer < arr.shape[1]):
        raise IndexError(f"Requested layer={layer}, array shape {arr.shape} in {p}")
    return arr[:, layer]


def load_xyz(block_dir: str, pos_dtype: str = "fp16") -> np.ndarray:
    ply = [f for f in os.listdir(block_dir) if f.endswith(".ply") and "original" in f]
    if not ply:
        raise FileNotFoundError(f"no *_original.ply in {block_dir}")
    ply_path = os.path.join(block_dir, ply[0])
    print(f"[read] ply: {ply_path}")
    pcd = read_pcdfile(ply_path)
    dt = np.float16 if pos_dtype == "fp16" else np.float32
    return np.asarray(pcd.points, dt)


def morton3D(xyz: np.ndarray) -> np.ndarray:
    pmin, pmax = xyz.min(0), xyz.max(0)
    span = np.maximum(pmax - pmin, 1e-9)
    u = (xyz - pmin) / span
    v = (u * (1 << 21)).astype(np.uint32)

    def split(a: np.ndarray) -> np.ndarray:
        a = (a | (a << 32)) & 0x1f00000000ffff
        a = (a | (a << 16)) & 0x1f0000ff0000ff
        a = (a | (a << 8))  & 0x100f00f00f00f00f
        a = (a | (a << 4))  & 0x10c30c30c30c30c3
        a = (a | (a << 2))  & 0x1249249249249249
        return a

    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return (split(x) | (split(y) << 1) | (split(z) << 2)).astype(np.uint64)


def build_seqs(xyz: np.ndarray, L: int = 96, stride: int = 64) -> List[Tuple[np.ndarray, int]]:
    key = morton3D(xyz)
    order = np.argsort(key, kind="mergesort")
    seqs: List[Tuple[np.ndarray, int]] = []
    n = len(order)
    for s in range(0, n, stride):
        e = min(s + L, n)
        idx = order[s:e]
        pad = L - len(idx)
        if pad > 0:
            pad_idx = np.full((pad,), idx[-1] if len(idx) > 0 else 0, dtype=idx.dtype)
            idx = np.concatenate([idx, pad_idx], axis=0)
        seqs.append((idx, pad))
    return seqs


def shard_writer(out_dir: str, shard_id: int, buf: Dict[str, np.ndarray], *, compress: bool = True) -> int:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"shard_{shard_id:06d}.npz")
    if compress:
        np.savez_compressed(path, **buf)
    else:
        np.savez(path, **buf)
    return shard_id + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="包含 block 子目录的根目录")
    ap.add_argument("--out", required=True, help="输出分片目录")
    ap.add_argument("--which", default="dc", choices=["dc", "sh"])
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--shard_size", type=int, default=32768)

    ap.add_argument("--index_dtype", default="int32", choices=["int32", "int64"])
    ap.add_argument("--pos_dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--no-compress", action="store_true", help="禁用 npz 压缩")
    ap.add_argument("--shift_target", action="store_true",
                    help="数据侧右移 target（用于非因果 Transformer）；最后一位与 pad 一起屏蔽")
    args = ap.parse_args()

    idx_dtype = np.int32 if args.index_dtype == "int32" else np.int64
    compress = (not args.no_compress)

    blocks = find_block_dirs(args.root)
    if not blocks:
        raise SystemExit(f"no block dirs in {args.root}")

    shard_id, num_samples = 0, 0

    for b in blocks:
        xyz = load_xyz(b, pos_dtype=args.pos_dtype)
        labels = load_indices(b, args.which, args.layer, dtype=idx_dtype)
        if labels.shape[0] != xyz.shape[0]:
            raise ValueError(f"size mismatch in {b}: xyz={xyz.shape[0]} vs labels={labels.shape[0]}")

        seqs = build_seqs(xyz, args.L, args.stride)

        buf = {
            "neighbor_indices": [],
            "target_indices": [],
            "positions": [],
            "key_padding_mask": []
        }

        for idx, pad in seqs:
            nb = labels[idx]
            tg = nb.copy()

            mask = np.ones(args.L, np.uint8)
            if pad > 0:
                mask[-pad:] = 0

            if args.shift_target:
                tg[:-1] = nb[1:]
                tg[-1] = 0
                mask[-1] = 0  # 屏蔽右移后最后一位

            buf["neighbor_indices"].append(nb)
            buf["target_indices"].append(tg)
            buf["positions"].append(xyz[idx])
            buf["key_padding_mask"].append(mask)

            if len(buf["neighbor_indices"]) == args.shard_size:
                shard_id = shard_writer(
                    args.out, shard_id, {k: np.array(v) for k, v in buf.items()}, compress=compress
                )
                for k in buf:
                    buf[k].clear()

        if buf["neighbor_indices"]:
            shard_id = shard_writer(
                args.out, shard_id, {k: np.array(v) for k, v in buf.items()}, compress=compress
            )

        num_samples += len(seqs)
        print(f"[ok] {b} -> {len(seqs)} seqs")

    manifest = {
        "root": os.path.abspath(args.root),
        "out": os.path.abspath(args.out),
        "which": args.which,
        "layer": args.layer,
        "L": args.L,
        "stride": args.stride,
        "shard_size": args.shard_size,
        "num_shards": shard_id,
        "num_samples": num_samples,
        "index_dtype": args.index_dtype,
        "pos_dtype": args.pos_dtype,
        "compress": compress,
        "shift_target": bool(args.shift_target)
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "iprior_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[done] samples={num_samples}, shards={shard_id}")


if __name__ == "__main__":
    main()