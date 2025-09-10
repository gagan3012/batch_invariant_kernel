#pragma once
#include <torch/extension.h>

void matmul_persistent_cuda(
    torch::Tensor const &a,
    torch::Tensor const &b,
    torch::Tensor &c,
    torch::Tensor const &bias);

void log_softmax_cuda(
    torch::Tensor const &input,
    torch::Tensor &output);

void mean_dim_cuda(
    torch::Tensor const &input,
    torch::Tensor &output,
    int dim);