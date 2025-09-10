#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cmath>

// Persistent matrix multiplication kernel
__global__ void matmul_kernel_persistent(
    const float *a_ptr,
    const float *b_ptr,
    float *c_ptr,
    const float *bias_ptr,
    int M, int N, int K,
    int stride_am, int stride_ak,
    int stride_bk, int stride_bn,
    int stride_cm, int stride_cn,
    int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K,
    int GROUP_SIZE_M, int NUM_SMS,
    bool HAS_BIAS)
{
    int start_pid = blockIdx.x;
    int num_pid_m = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    int num_pid_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    int k_tiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    int num_tiles = num_pid_m * num_pid_n;

    int num_pid_in_group = GROUP_SIZE_M * num_pid_n;

    for (int tile_id = start_pid; tile_id < num_tiles; tile_id += NUM_SMS)
    {
        int group_id = tile_id / num_pid_in_group;
        int first_pid_m = group_id * GROUP_SIZE_M;
        int group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M);
        int pid_m = first_pid_m + (tile_id % group_size_m);
        int pid_n = (tile_id % num_pid_in_group) / group_size_m;

        int start_m = pid_m * BLOCK_SIZE_M;
        int start_n = pid_n * BLOCK_SIZE_N;

        // Shared memory for tile computation
        __shared__ float As[16][16]; // Adjust size based on BLOCK_SIZE
        __shared__ float Bs[16][16];

        float accumulator = 0.0f;
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        // Bounds checking
        if (start_m + tx < M && start_n + ty < N)
        {
            // K-dimension loop
            for (int ki = 0; ki < k_tiles; ki++)
            {
                int k_start = ki * BLOCK_SIZE_K;

                // Load tiles into shared memory
                if (k_start + tx < K && start_m + ty < M)
                {
                    As[ty][tx] = a_ptr[(start_m + ty) * stride_am + (k_start + tx) * stride_ak];
                }
                else
                {
                    As[ty][tx] = 0.0f;
                }

                if (k_start + ty < K && start_n + tx < N)
                {
                    Bs[ty][tx] = b_ptr[(k_start + ty) * stride_bk + (start_n + tx) * stride_bn];
                }
                else
                {
                    Bs[ty][tx] = 0.0f;
                }

                __syncthreads();

                // Compute partial dot product
                for (int k = 0; k < min(BLOCK_SIZE_K, K - k_start); k++)
                {
                    accumulator += As[ty][k] * Bs[k][tx];
                }

                __syncthreads();
            }

            // Add bias if present
            if (HAS_BIAS && bias_ptr != nullptr)
            {
                accumulator += bias_ptr[start_n + tx];
            }

            // Store result
            c_ptr[(start_m + ty) * stride_cm + (start_n + tx) * stride_cn] = accumulator;
        }
    }
}

// Log softmax kernel
__global__ void log_softmax_kernel(
    const float *input_ptr,
    float *output_ptr,
    int input_row_stride,
    int output_row_stride,
    int n_cols,
    int BLOCK_SIZE)
{
    int row_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Find maximum value in the row for numerical stability
    __shared__ float max_val;
    __shared__ float sum_exp;

    if (tid == 0)
    {
        max_val = -INFINITY;
        sum_exp = 0.0f;
    }
    __syncthreads();

    // Reduction to find max
    float thread_max = -INFINITY;
    for (int col = tid; col < n_cols; col += blockDim.x)
    {
        float val = input_ptr[row_idx * input_row_stride + col];
        thread_max = fmaxf(thread_max, val);
    }

    // Block-wide reduction for max
    __shared__ float sdata[256];
    sdata[tid] = thread_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        max_val = sdata[0];
    }
    __syncthreads();

    // Compute sum of exp(x - max_val)
    float thread_sum = 0.0f;
    for (int col = tid; col < n_cols; col += blockDim.x)
    {
        float val = input_ptr[row_idx * input_row_stride + col];
        thread_sum += expf(val - max_val);
    }

    // Block-wide reduction for sum
    sdata[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        sum_exp = sdata[0];
    }
    __syncthreads();

    float log_sum_exp = logf(sum_exp);

    // Compute final log_softmax values
    for (int col = tid; col < n_cols; col += blockDim.x)
    {
        float val = input_ptr[row_idx * input_row_stride + col];
        output_ptr[row_idx * output_row_stride + col] = val - max_val - log_sum_exp;
    }
}

// Mean reduction kernel
__global__ void mean_kernel(
    const float *input_ptr,
    float *output_ptr,
    int input_stride0, int input_stride1, int input_stride2,
    int output_stride0, int output_stride1,
    int M, int N, int K,
    int BLOCK_SIZE)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;

    if (pid >= M * K)
        return;

    int m_idx = pid / K;
    int k_idx = pid % K;

    float acc = 0.0f;
    for (int n = 0; n < N; n++)
    {
        int input_idx = m_idx * input_stride0 + n * input_stride1 + k_idx * input_stride2;
        acc += input_ptr[input_idx];
    }

    float mean_val = acc / N;
    int output_idx = m_idx * output_stride0 + k_idx * output_stride1;
    output_ptr[output_idx] = mean_val;
}

// Host functions that launch the kernels
void matmul_persistent_cuda(
    torch::Tensor const &a,
    torch::Tensor const &b,
    torch::Tensor &c,
    torch::Tensor const &bias)
{
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int NUM_SMS = prop.multiProcessorCount;

    // Block sizes
    const int BLOCK_SIZE_M = 128;
    const int BLOCK_SIZE_N = 128;
    const int BLOCK_SIZE_K = 64;
    const int GROUP_SIZE_M = 8;

    // Grid configuration
    const int num_pid_m = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    const int num_pid_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    const int grid_size = min(NUM_SMS, num_pid_m * num_pid_n);

    dim3 block(16, 16);
    dim3 grid_dim(grid_size);

    matmul_kernel_persistent<<<grid_dim, block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        GROUP_SIZE_M, NUM_SMS,
        bias.defined());
}

void log_softmax_cuda(
    torch::Tensor const &input,
    torch::Tensor &output)
{
    const auto original_shape = input.sizes();
    auto input_2d = input.reshape({-1, input.size(-1)}).contiguous();
    auto output_2d = output.reshape({-1, output.size(-1)});

    const int n_rows = input_2d.size(0);
    const int n_cols = input_2d.size(1);

    const int BLOCK_SIZE = 256;

    log_softmax_kernel<<<n_rows, BLOCK_SIZE>>>(
        input_2d.data_ptr<float>(),
        output_2d.data_ptr<float>(),
        input_2d.stride(0),
        output_2d.stride(0),
        n_cols,
        BLOCK_SIZE);
}

void mean_dim_cuda(
    torch::Tensor const &input,
    torch::Tensor &output,
    int dim)
{
    auto shape = input.sizes().vec();

    int M = 1;
    for (int i = 0; i < dim; i++)
    {
        M *= shape[i];
    }

    int N = shape[dim];

    int K = 1;
    for (int i = dim + 1; i < shape.size(); i++)
    {
        K *= shape[i];
    }

    auto input_3d = input.reshape({M, N, K});
    auto output_2d = output.reshape({M, K});

    const int BLOCK_SIZE = 256;
    const int grid_size = (M * K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    mean_kernel<<<grid_size, BLOCK_SIZE>>>(
        input_3d.data_ptr<float>(),
        output_2d.data_ptr<float>(),
        input_3d.stride(0), input_3d.stride(1), input_3d.stride(2),
        output_2d.stride(0), output_2d.stride(1),
        M, N, K,
        BLOCK_SIZE);
}