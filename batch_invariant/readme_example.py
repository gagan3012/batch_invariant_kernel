# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "numpy",
#     "kernels",
# ]
# ///

import torch
from kernels import get_kernel

# Load batch_invariant_kernel via kernels library
batch_invariant_kernel = get_kernel("gagan3012/batch_invariant_kernel")

# Set device and seed for reproducibility
device = "cuda"
torch.manual_seed(42)
torch.cuda.manual_seed(42)

print("🚀 Testing batch_invariant_kernel from Hugging Face Hub")
print(f"✅ CUDA is available. Using device: {torch.cuda.get_device_name()}")

# Test 1: Matrix Multiplication
print("\n" + "=" * 60)
print("🧪 Test 1: Persistent Matrix Multiplication")
print("=" * 60)

# Parameters for matrix multiplication
M, K, N = 512, 256, 1024
a = torch.randn(M, K, device=device, dtype=torch.float32)
b = torch.randn(K, N, device=device, dtype=torch.float32)
bias = torch.randn(N, device=device, dtype=torch.float32)

print(f"Matrix A shape: {a.shape}")
print(f"Matrix B shape: {b.shape}")
print(f"Bias shape: {bias.shape}")

# Run matrix multiplication without bias
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
output_no_bias = batch_invariant_kernel.matmul_persistent(a, b)
end_event.record()
torch.cuda.synchronize()
time_no_bias = start_event.elapsed_time(end_event)

print(f"\nMatrix multiplication (no bias) completed!")
print(f"Output shape: {output_no_bias.shape}")
print(f"Execution time: {time_no_bias:.3f} ms")

# Run matrix multiplication with bias
start_event.record()
output_with_bias = batch_invariant_kernel.matmul_persistent(a, b, bias)
end_event.record()
torch.cuda.synchronize()
time_with_bias = start_event.elapsed_time(end_event)

print(f"\nMatrix multiplication (with bias) completed!")
print(f"Output shape: {output_with_bias.shape}")
print(f"Execution time: {time_with_bias:.3f} ms")

# Verify correctness
expected_no_bias = torch.mm(a, b)
expected_with_bias = torch.mm(a, b) + bias

max_diff_no_bias = torch.max(torch.abs(output_no_bias - expected_no_bias)).item()
max_diff_with_bias = torch.max(torch.abs(output_with_bias - expected_with_bias)).item()

print(f"Max difference (no bias): {max_diff_no_bias:.6f}")
print(f"Max difference (with bias): {max_diff_with_bias:.6f}")

# Test 2: Log Softmax
print("\n" + "=" * 60)
print("🧪 Test 2: Log Softmax")
print("=" * 60)

# Parameters for log softmax (typical attention dimensions)
batch_size = 4
seq_len = 512
vocab_size = 32000

logits = torch.randn(
    batch_size, seq_len, vocab_size, device=device, dtype=torch.float32
)
print(f"Input logits shape: {logits.shape}")

# Run log softmax
start_event.record()
log_probs = batch_invariant_kernel.log_softmax(logits, dim=-1)
end_event.record()
torch.cuda.synchronize()
time_log_softmax = start_event.elapsed_time(end_event)

print(f"\nLog softmax completed!")
print(f"Output shape: {log_probs.shape}")
print(f"Execution time: {time_log_softmax:.3f} ms")

# Verify correctness
expected_log_probs = torch.log_softmax(logits, dim=-1)
max_diff_log_softmax = torch.max(torch.abs(log_probs - expected_log_probs)).item()
print(f"Max difference vs PyTorch: {max_diff_log_softmax:.6f}")

# Test 3: Mean Reduction
print("\n" + "=" * 60)
print("🧪 Test 3: Mean Dimension Reduction")
print("=" * 60)

# Parameters for mean reduction (typical layer norm dimensions)
batch_size = 8
seq_len = 256
hidden_size = 768

hidden_states = torch.randn(
    batch_size, seq_len, hidden_size, device=device, dtype=torch.float32
)
print(f"Input hidden states shape: {hidden_states.shape}")

# Test reduction along different dimensions
for dim in [0, 1, 2]:
    start_event.record()
    mean_output = batch_invariant_kernel.mean_dim(hidden_states, dim=dim, keepdim=False)
    end_event.record()
    torch.cuda.synchronize()
    time_mean = start_event.elapsed_time(end_event)

    expected_mean = torch.mean(hidden_states, dim=dim, keepdim=False)
    max_diff_mean = torch.max(torch.abs(mean_output - expected_mean)).item()

    print(f"\nMean reduction along dim {dim}:")
    print(f"  Output shape: {mean_output.shape}")
    print(f"  Execution time: {time_mean:.3f} ms")
    print(f"  Max difference vs PyTorch: {max_diff_mean:.6f}")

# Test 4: End-to-End Attention-like Computation
print("\n" + "=" * 60)
print("🧪 Test 4: End-to-End Attention-like Computation")
print("=" * 60)

# Simulate a simple attention computation using our kernels
batch_size = 4
seq_len = 128
hidden_size = 512
num_heads = 8
head_dim = hidden_size // num_heads

# Input embeddings
x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=torch.float32)

# Weight matrices for Q, K, V projections
w_q = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32)
w_k = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32)
w_v = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32)
w_o = torch.randn(hidden_size, hidden_size, device=device, dtype=torch.float32)

print(f"Input shape: {x.shape}")
print("Computing Q, K, V projections using batch_invariant matmul...")

# Reshape for batch matrix multiplication
x_flat = x.view(-1, hidden_size)  # (batch_size * seq_len, hidden_size)

start_event.record()

# Compute Q, K, V using our custom matmul
q_flat = batch_invariant_kernel.matmul_persistent(x_flat, w_q)
k_flat = batch_invariant_kernel.matmul_persistent(x_flat, w_k)
v_flat = batch_invariant_kernel.matmul_persistent(x_flat, w_v)

# Reshape to multi-head format
q = q_flat.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
k = k_flat.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
v = v_flat.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

# Compute attention scores
scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim**0.5)

# Apply softmax using our custom log_softmax (convert to softmax)
log_attn_weights = batch_invariant_kernel.log_softmax(scores, dim=-1)
attn_weights = torch.exp(log_attn_weights)

# Apply attention to values
attn_output = torch.matmul(attn_weights, v)

# Reshape and apply output projection
attn_output = (
    attn_output.transpose(1, 2).contiguous().view(batch_size * seq_len, hidden_size)
)
final_output = batch_invariant_kernel.matmul_persistent(attn_output, w_o)
final_output = final_output.view(batch_size, seq_len, hidden_size)

end_event.record()
torch.cuda.synchronize()
total_time = start_event.elapsed_time(end_event)

print(f"\nEnd-to-end attention computation completed!")
print(f"Final output shape: {final_output.shape}")
print(f"Total execution time: {total_time:.3f} ms")
print(
    f"Output tensor stats - Mean: {final_output.mean().item():.4f}, Std: {final_output.std().item():.4f}"
)

