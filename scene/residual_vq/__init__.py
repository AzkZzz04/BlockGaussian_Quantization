# scene/residual_vq/__init__.py
# Unified RVQ interface - Quantize_RVQ handles both single-layer (K-Means) and multi-layer RVQ
from .rvq_adapter import Quantize_RVQ
from .rvq_apply import rvq_apply, rvq_apply_dict
from .rvq_utils import save_rvq_artifacts, load_rvq_codebooks, load_manifest, validate_codebooks_against_manifest

# For advanced users who need direct access to internals
from .vq_module import Quantize_kMeans
from .rvq import ResidualKMeansVQ

__all__ = [
    # Primary interface
    "Quantize_RVQ",
    
    # Pure functions (runtime only)
    "rvq_apply", "rvq_apply_dict",
    
    # I/O utilities
    "save_rvq_artifacts", "load_rvq_codebooks", "load_manifest", "validate_codebooks_against_manifest",
    
    # Advanced/internal use
    "Quantize_kMeans", "ResidualKMeansVQ",
]