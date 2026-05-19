#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "dimq_common.cuh"

namespace dimq {
namespace {

template <typename scalar_t>
__global__ void build_product_table_kernel(const scalar_t* __restrict__ codebook,
                                           float act_scale,
                                           int act_zp,
                                           float* __restrict__ table) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 256) {
    return;
  }
  int w_idx = idx / 16;
  int a_idx = idx % 16;
  float w = scalar_to_float(codebook[w_idx]);
  float a = act_scale * static_cast<float>(a_idx - act_zp);
  table[idx] = w * a;
}

}  // namespace

torch::Tensor build_product_table_cuda(torch::Tensor codebook, double act_scale, int64_t act_zp) {
  TORCH_CHECK(codebook.is_cuda(), "build_product_table expects a CUDA tensor");
  TORCH_CHECK(codebook.is_floating_point(), "build_product_table expects floating codebook");
  TORCH_CHECK(codebook.numel() == 16, "build_product_table expects exactly 16 codebook entries");
  TORCH_CHECK(act_scale > 0.0, "act_scale must be positive");
  TORCH_CHECK(act_zp >= 0 && act_zp <= 15, "act_zero_point must be in [0, 15]");
  auto contiguous = codebook.contiguous().view({16});
  auto table = torch::empty({16, 16}, contiguous.options().dtype(torch::kFloat32));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(contiguous.scalar_type(), "build_product_table_cuda", [&] {
    build_product_table_kernel<scalar_t><<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        contiguous.data_ptr<scalar_t>(),
        static_cast<float>(act_scale),
        static_cast<int>(act_zp),
        table.data_ptr<float>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return table;
}

}  // namespace dimq
