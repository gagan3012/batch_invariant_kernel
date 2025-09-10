#include <torch/extension.h>
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)
{
    ops.def("matmul_persistent(Tensor a, Tensor b, Tensor! c, Tensor? bias) -> ()");
    ops.def("log_softmax(Tensor input, Tensor! output) -> ()");
    ops.def("mean_dim(Tensor input, Tensor! output, int dim) -> ()");

    ops.impl("matmul_persistent", torch::kCUDA, &matmul_persistent_cuda);
    ops.impl("log_softmax", torch::kCUDA, &log_softmax_cuda);
    ops.impl("mean_dim", torch::kCUDA, &mean_dim_cuda);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)