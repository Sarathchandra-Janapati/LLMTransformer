"""Small attention components that expose their intermediate tensors."""

from .lab import (
    AttentionLabExample,
    TASK_MODES,
    build_attention_lab_example,
    build_i_love_transformers_example,
    sinusoidal_position_encodings,
    task_attention_mask,
)
from .multi_head import (
    MultiHeadAttentionCache,
    MultiHeadAttentionGradients,
    MultiHeadAttentionResult,
    MultiHeadSelfAttention,
)
from .single_head import (
    AttentionCache,
    AttentionGradients,
    AttentionResult,
    SingleHeadSelfAttention,
    causal_attention_mask,
    softmax,
)

__all__ = [
    "AttentionCache",
    "AttentionGradients",
    "AttentionLabExample",
    "AttentionResult",
    "MultiHeadAttentionCache",
    "MultiHeadAttentionGradients",
    "MultiHeadAttentionResult",
    "MultiHeadSelfAttention",
    "SingleHeadSelfAttention",
    "TASK_MODES",
    "build_attention_lab_example",
    "build_i_love_transformers_example",
    "causal_attention_mask",
    "sinusoidal_position_encodings",
    "softmax",
    "task_attention_mask",
]
