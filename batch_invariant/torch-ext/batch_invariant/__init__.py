import torch
import torch.nn as nn
import math
from ._ops import ops


def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor = None
) -> torch.Tensor:
    """
    Persistent matrix multiplication with optional bias.

    Args:
        a: Input tensor of shape (M, K)
        b: Input tensor of shape (K, N)
        bias: Optional bias tensor of shape (N,)

    Returns:
        Output tensor of shape (M, N)
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    assert bias is None or bias.dim() == 1, "Bias must be 1D"

    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    ops.matmul_persistent(a, b, c, bias)

    return c


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute log_softmax using custom CUDA kernel.

    Args:
        input: Input tensor
        dim: Dimension along which to compute log_softmax (only -1 supported)

    Returns:
        Tensor with log_softmax applied
    """
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError(
            "This implementation only supports log_softmax along the last dimension"
        )

    output = torch.empty_like(input)
    ops.log_softmax(input, output)

    return output


def mean_dim(
    input: torch.Tensor, dim: int, keepdim: bool = False, dtype: torch.dtype = None
) -> torch.Tensor:
    """
    Compute mean along a single dimension.

    Args:
        input: Input tensor
        dim: Single dimension along which to compute mean
        keepdim: Whether to keep the reduced dimension
        dtype: Output dtype

    Returns:
        Tensor with mean values along specified dimension
    """
    assert input.is_cuda, "Input must be a CUDA tensor"
    assert -input.ndim <= dim < input.ndim, f"Invalid dimension {dim}"

    if dim < 0:
        dim = dim + input.ndim

    if dtype is None:
        if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
            dtype = torch.float32
        else:
            dtype = input.dtype

    if input.dtype != dtype:
        input = input.to(dtype)

    shape = list(input.shape)

    if keepdim:
        output_shape = shape.copy()
        output_shape[dim] = 1
    else:
        output_shape = shape[:dim] + shape[dim + 1 :]

    output = torch.empty(output_shape, dtype=dtype, device=input.device)
    ops.mean_dim(input, output, dim)

    return output


# Batch invariant mode functionality (if you still want the mode switching)
def mm_batch_invariant(a, b):
    return matmul_persistent(a, b)


def addmm_batch_invariant(bias, a, b):
    return matmul_persistent(a, b, bias=bias)


def _log_softmax_batch_invariant(input, dim, _half_to_float):
    assert not _half_to_float, "not implemented"
    return log_softmax(input, dim=dim)


def mean_batch_invariant(input, dim, keepdim=False, dtype: torch.dtype = None):
    if len(dim) == 1:
        return mean_dim(input, dim[0], keepdim=keepdim, dtype=dtype)
    else:
        # Multi-dimensional mean fallback
        n_elems = 1
        for d in dim:
            n_elems *= input.shape[d]
        return torch.sum(input, dim=dim, keepdim=keepdim, dtype=torch.float32) / n_elems

class BatchInvariantAttention(nn.Module):
    """
    Batch invariant multi-head attention implementation.
    Compatible with transformers library integration.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 2048)

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Linear projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.Tensor = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V using batch invariant matrix multiplication
        query_states = self._batch_invariant_linear(hidden_states, self.q_proj.weight)
        key_states = self._batch_invariant_linear(hidden_states, self.k_proj.weight)
        value_states = self._batch_invariant_linear(hidden_states, self.v_proj.weight)

        # Reshape for multi-head attention
        query_states = query_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax using batch invariant log_softmax
        attn_weights_log = log_softmax(attn_weights, dim=-1)
        attn_weights = torch.exp(attn_weights_log)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self._batch_invariant_linear(attn_output, self.o_proj.weight)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (past_key_value,)

        return outputs

    def _batch_invariant_linear(
        self, input_tensor: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Apply linear transformation using batch invariant matrix multiplication"""
        original_shape = input_tensor.shape
        input_2d = input_tensor.view(-1, original_shape[-1])
        output_2d = matmul_persistent(input_2d, weight.t())
        return output_2d.view(*original_shape[:-1], -1)


class BatchInvariantMLP(nn.Module):
    """
    Batch invariant MLP implementation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = (
            nn.SiLU()
        )  # or whatever activation function is specified in config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use batch invariant matrix multiplication for projections
        gate = self._batch_invariant_linear(x, self.gate_proj.weight)
        up = self._batch_invariant_linear(x, self.up_proj.weight)

        # Apply activation
        intermediate = self.act_fn(gate) * up

        # Down projection
        output = self._batch_invariant_linear(intermediate, self.down_proj.weight)
        return output

    def _batch_invariant_linear(
        self, input_tensor: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Apply linear transformation using batch invariant matrix multiplication"""
        original_shape = input_tensor.shape
        input_2d = input_tensor.view(-1, original_shape[-1])
        output_2d = matmul_persistent(input_2d, weight.t())
        return output_2d.view(*original_shape[:-1], -1)


class BatchInvariantRMSNorm(nn.Module):
    """
    Batch invariant RMS normalization implementation.
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        # Compute mean square using batch invariant mean
        variance = mean_dim(hidden_states.pow(2), dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


# Export the layer classes
__all__ += ["BatchInvariantAttention", "BatchInvariantMLP", "BatchInvariantRMSNorm"]