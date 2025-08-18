#!/usr/bin/env python3
"""
VQ Decompression Script for BlockGaussian
Convert quantized PLY (with VQ codebooks) to uncompressed PLY for SIBR or other tools.
"""

import os
import yaml
import torch
import numpy as np
import easydict
from tqdm import tqdm
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement


# ------------------------------ Utils ------------------------------

def parse_cfg(args) -> easydict.EasyDict:
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"config does not exist: {args.config}")
    with open(args.config, "rb") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if getattr(args, "scene_dirpath", None) is not None:
        cfg["scene_dirpath"] = args.scene_dirpath
    if getattr(args, "output_dirpath", None) is not None:
        cfg["output_dirpath"] = args.output_dirpath
    return easydict.EasyDict(cfg)


def _as_float32(x):
    return np.asarray(x, dtype=np.float32)


def _take_first_n(arr, n):
    """Safely truncate an array-like (numpy or torch) to first n elements."""
    if isinstance(arr, torch.Tensor):
        return arr[:n]
    return arr[:n]


# ------------------------------ Core ------------------------------

def decompress_ply_with_vq(ply_path: str, codebook_path: str, output_path: str, sh_degree: int = 3):
    """
    Decompress a single PLY using VQ codebooks and write an uncompressed PLY.
    """
    print(f"Loading compressed PLY: {ply_path}")
    print(f"Loading VQ codebooks:   {codebook_path}")

    if not os.path.exists(codebook_path):
        raise FileNotFoundError(f"Codebook file not found: {codebook_path}")

    codebooks = torch.load(codebook_path, map_location="cpu")
    print(f"Loaded codebooks with keys: {list(codebooks.keys())}")

    plydata = PlyData.read(ply_path)
    vertices = plydata["vertex"]

    # --- Read base attributes ---
    xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]])
    opacity = np.asarray(vertices["opacity"])
    scaling = np.column_stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]])
    rotation = np.column_stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]])

    # --- Read compressed SH features (DC + rest) as stored in the PLY ---
    f_dc = np.column_stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]])

    rest_props = [p for p in vertices.data.dtype.names if p.startswith("f_rest_")]
    rest_props = sorted(rest_props, key=lambda x: int(x.split("_")[-1]))
    print(f"Found {len(rest_props)} rest feature properties")
    f_rest = np.column_stack([vertices[p] for p in rest_props]) if len(rest_props) > 0 else np.zeros((len(vertices), 0), dtype=np.float32)
    if f_rest.size:
        print(f"Rest features shape: {f_rest.shape}")

    N = xyz.shape[0]

    # --- Defaults: copy through if no codebook ---
    decompressed_dc = f_dc.copy()
    decompressed_rest = f_rest.copy()

    # --- Decompress DC using VQ codebook ---
    if "dc_centers" in codebooks:
        print("Decompressing DC features...")
        dc_centers = codebooks["dc_centers"]  # [K, 3]
        
        # Check if we have cluster IDs in codebook or need to use PLY data
        if "dc_cluster_ids" in codebooks:
            dc_ids = codebooks["dc_cluster_ids"]
        else:
            # Use the indices stored in PLY (they should be the same)
            dc_ids = f_dc.astype(np.int32)
        
        if len(dc_ids) != N:
            print(f"  Note: dc_cluster_ids len {len(dc_ids)} != points {N}, truncating to {N}")
            dc_ids = _take_first_n(dc_ids, N)
        
        # Index centers
        if isinstance(dc_centers, torch.Tensor):
            dd = dc_centers[dc_ids].detach().cpu().numpy()
        else:
            dd = np.asarray(dc_centers)[np.asarray(dc_ids)]
        decompressed_dc = dd.astype(np.float32)
        print(f"  DC features decompressed: {decompressed_dc.shape}")

    # --- Decompress Rest using VQ codebook ---
    if "rest_centers" in codebooks and f_rest.size:
        print("Decompressing Rest features...")
        rest_centers = codebooks["rest_centers"]  # [K, D]
        
        # Check if we have cluster IDs in codebook or need to use PLY data
        if "rest_cluster_ids" in codebooks:
            rest_ids = codebooks["rest_cluster_ids"]
        else:
            # Use the indices stored in PLY
            rest_ids = f_rest.astype(np.int32)
        
        if len(rest_ids) != N:
            print(f"  Note: rest_cluster_ids len {len(rest_ids)} != points {N}, truncating to {N}")
            rest_ids = _take_first_n(rest_ids, N)
        
        # Index centers
        if isinstance(rest_centers, torch.Tensor):
            rr = rest_centers[rest_ids].detach().cpu().numpy()
        else:
            rr = np.asarray(rest_centers)[np.asarray(rest_ids)]
        decompressed_rest = rr.astype(np.float32)
        print(f"  Rest features decompressed: {decompressed_rest.shape}")

    # --- Write decompressed PLY ---
    print(f"Writing decompressed PLY to: {output_path}")
    
    # Calculate RGB colors from decompressed DC features
    rgb = decompressed_dc.reshape(-1, 3)
    # Apply sigmoid and scale to 0-255
    rgb = torch.sigmoid(torch.from_numpy(rgb)).numpy()
    rgb = (rgb * 255).astype(np.uint8)
    
    # Build PLY header - standard visualization format
    attr_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'opacity']
    attr_types = ['f4', 'f4', 'f4', 'u1', 'u1', 'u1', 'f4']
    
    # Add scale and rotation attributes
    for i in range(scaling.shape[1]):
        attr_names.append(f'scale_{i}')
        attr_types.append('f4')
    
    for i in range(rotation.shape[1]):
        attr_names.append(f'rot_{i}')
        attr_types.append('f4')
    
    # Add decompressed features
    for i in range(decompressed_dc.shape[1]):
        attr_names.append(f'f_dc_{i}')
        attr_types.append('f4')
    
    for i in range(decompressed_rest.shape[1]):
        attr_names.append(f'f_rest_{i}')
        attr_types.append('f4')
    
    # Use PlyData for reliable writing
    dtype_full = [(name, dtype) for name, dtype in zip(attr_names, attr_types)]
    elements = np.empty(len(xyz), dtype=dtype_full)
    
    # Fill data
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red'] = rgb[:, 0]
    elements['green'] = rgb[:, 1]
    elements['blue'] = rgb[:, 2]
    elements['opacity'] = opacity.flatten()
    
    for i in range(scaling.shape[1]):
        elements[f'scale_{i}'] = scaling[:, i]
    
    for i in range(rotation.shape[1]):
        elements[f'rot_{i}'] = rotation[:, i]
    
    for i in range(decompressed_dc.shape[1]):
        elements[f'f_dc_{i}'] = decompressed_dc[:, i]
    
    for i in range(decompressed_rest.shape[1]):
        elements[f'f_rest_{i}'] = decompressed_rest[:, i]
    
    # Create PlyElement and write
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(output_path)
    
    print(f"✅ Decompressed PLY saved to {output_path} with {len(xyz)} points")


def batch_decompress_vq(cfg):
    """Batch decompress all VQ-compressed PLY files in the output directory."""
    output_dir = cfg.output_dirpath
    
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist: {output_dir}")
        return
    
    # Find all compressed PLY files
    compressed_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith("_compressed.ply"):
                compressed_files.append(os.path.join(root, file))
    
    if not compressed_files:
        print("No compressed PLY files found.")
        return
    
    print(f"Found {len(compressed_files)} compressed PLY files to decompress.")
    
    # Process each compressed file
    for compressed_ply in tqdm(compressed_files, desc="Decompressing"):
        try:
            # Find corresponding codebook
            base_name = compressed_ply.replace("_compressed.ply", "")
            codebook_path = f"{base_name}_vq_codebooks.pth"
            
            if not os.path.exists(codebook_path):
                print(f"Warning: Codebook not found for {compressed_ply}")
                continue
            
            # Generate output path
            output_path = compressed_ply.replace("_compressed.ply", "_decompressed.ply")
            
            # Decompress
            decompress_ply_with_vq(compressed_ply, codebook_path, output_path)
            
        except Exception as e:
            print(f"Error processing {compressed_ply}: {e}")
            continue
    
    print("Batch decompression complete!")


# ------------------------------ Main ------------------------------

if __name__ == "__main__":
    parser = ArgumentParser(description="VQ Decompression for BlockGaussian")
    parser.add_argument("--config", "-c", type=str, required=True, help="Config file path")
    parser.add_argument("--ply", "-p", type=str, help="Single compressed PLY file to decompress")
    parser.add_argument("--codebook", "-k", type=str, help="Codebook file path (if different from default)")
    parser.add_argument("--output", "-o", type=str, help="Output PLY path (if different from default)")
    parser.add_argument("--batch", "-b", action="store_true", help="Batch decompress all compressed PLY files")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode
        cfg = parse_cfg(args)
        batch_decompress_vq(cfg)
    elif args.ply:
        # Single file mode
        if not args.codebook:
            # Try to find codebook automatically
            base_name = args.ply.replace("_compressed.ply", "")
            args.codebook = f"{base_name}_vq_codebooks.pth"
        
        if not args.output:
            args.output = args.ply.replace("_compressed.ply", "_decompressed.ply")
        
        decompress_ply_with_vq(args.ply, args.codebook, args.output)
    else:
        print("Please specify either --batch or --ply")
        parser.print_help()