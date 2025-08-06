"""
JAX-MXFP4: High-performance 4-bit quantization for JAX.
"""

from .quantize import (
    quantize_to_mxfp4,
    dequantize_mxfp4,
    quantize_model_weights,
    dynamic_quantize_activations,
    analyze_quantization_error,
)

from .layers import (
    MXFP4Linear,
    MXFP4Embedding,
    MXFP4Attention,
    MXFP4MLP,
)

from .kernels import (
    mxfp4_matmul,
    mxfp4_matmul_batched,
    fused_mxfp4_gelu,
    fused_mxfp4_layernorm,
    get_optimized_kernel,
)

# Model and training modules are planned for future releases
# from .model import (
#     MXFP4Model,
#     MXFP4Config,
#     convert_model_to_mxfp4,
#     load_mxfp4_weights,
#     save_mxfp4_weights,
# )

# from .training import (
#     MXFP4Trainer,
#     create_mxfp4_optimizer,
#     quantization_aware_training,
# )

__version__ = "0.1.0"

__all__ = [
    # Quantization functions
    "quantize_to_mxfp4",
    "dequantize_mxfp4",
    "quantize_model_weights",
    "dynamic_quantize_activations",
    "analyze_quantization_error",
    # Layers
    "MXFP4Linear",
    "MXFP4Embedding",
    "MXFP4Attention",
    "MXFP4MLP",
    # Kernels
    "mxfp4_matmul",
    "mxfp4_matmul_batched",
    "fused_mxfp4_gelu",
    "fused_mxfp4_layernorm",
    "get_optimized_kernel",
    # Model - commented out until implemented
    # "MXFP4Model",
    # "MXFP4Config",
    # "convert_model_to_mxfp4",
    # "load_mxfp4_weights",
    # "save_mxfp4_weights",
    # Training - commented out until implemented
    # "MXFP4Trainer",
    # "create_mxfp4_optimizer",
    # "quantization_aware_training",
]