#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <vector>

#include "dimq_common.cuh"

namespace dimq {

torch::Tensor build_product_table_cuda(torch::Tensor codebook, double act_scale, int64_t act_zp);

namespace {

template <typename scalar_t>
__global__ void dimq_linear_lut_kernel(const scalar_t* __restrict__ x,
                                       const uint8_t* __restrict__ packed_w,
                                       const float* __restrict__ product_table,
                                       const float* __restrict__ bias,
                                       float* __restrict__ y,
                                       int64_t M,
                                       int64_t N,
                                       int64_t K,
                                       float act_scale,
                                       int act_zp,
                                       bool has_bias) {
  __shared__ float table_s[256];
  int local = threadIdx.y * blockDim.x + threadIdx.x;
  int block_threads = blockDim.x * blockDim.y;
  for (int i = local; i < 256; i += block_threads) {
    table_s[i] = product_table[i];
  }
  __syncthreads();

  int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t m = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) {
    return;
  }

  float acc = 0.0f;
  int64_t x_base = m * K;
  int64_t w_base = n * K;
  for (int64_t k = 0; k < K; ++k) {
    uint8_t a_idx = quantize_u4(scalar_to_float(x[x_base + k]), act_scale, act_zp);
    uint8_t w_idx = get_u4(packed_w, w_base + k);
    acc += table_s[static_cast<int>(w_idx) * 16 + static_cast<int>(a_idx)];
  }
  if (has_bias) {
    acc += bias[n];
  }
  y[m * N + n] = acc;
}

}  // namespace

torch::Tensor linear_lut_cuda(torch::Tensor x,
                              torch::Tensor packed_weight,
                              torch::Tensor codebook,
                              double act_scale,
                              int64_t act_zp,
                              c10::optional<torch::Tensor> bias,
                              int64_t out_features,
                              int64_t in_features) {
  TORCH_CHECK(x.is_cuda(), "linear_lut expects CUDA input");
  TORCH_CHECK(packed_weight.is_cuda(), "linear_lut expects CUDA packed_weight");
  TORCH_CHECK(codebook.is_cuda(), "linear_lut expects CUDA codebook");
  TORCH_CHECK(x.is_floating_point(), "linear_lut expects floating input");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8, "packed_weight must be uint8");
  TORCH_CHECK(codebook.is_floating_point(), "codebook must be floating point");
  TORCH_CHECK(codebook.numel() == 16, "codebook must have 16 entries for W4");
  TORCH_CHECK(act_scale > 0.0, "act_scale must be positive");
  TORCH_CHECK(act_zp >= 0 && act_zp <= 15, "act_zero_point must be in [0, 15]");
  TORCH_CHECK(out_features > 0 && in_features > 0, "linear dimensions must be positive");
  TORCH_CHECK(x.numel() % in_features == 0, "input numel must be divisible by in_features");
  TORCH_CHECK(packed_weight.numel() >= (out_features * in_features + 1) / 2,
              "packed_weight is too small for out_features * in_features");
  TORCH_CHECK(x.size(-1) == in_features, "input last dimension must equal in_features");

  auto x_contig = x.contiguous();
  auto packed = packed_weight.contiguous().view({-1});
  auto table = build_product_table_cuda(codebook, act_scale, act_zp).contiguous().view({256});

  c10::optional<torch::Tensor> bias_float = c10::nullopt;
  const float* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda(), "bias must be CUDA when provided");
    TORCH_CHECK(bias->numel() == out_features, "bias numel must equal out_features");
    bias_float = bias->contiguous().to(torch::kFloat32).view({out_features});
    bias_ptr = bias_float->data_ptr<float>();
  }

  int64_t M = x_contig.numel() / in_features;
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = out_features;
  auto y = torch::empty({M, out_features}, x_contig.options().dtype(torch::kFloat32));

  dim3 block(16, 16);
  dim3 grid((out_features + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_contig.scalar_type(), "linear_lut_cuda", [&] {
    dimq_linear_lut_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x_contig.data_ptr<scalar_t>(),
        packed.data_ptr<uint8_t>(),
        table.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        M,
        out_features,
        in_features,
        static_cast<float>(act_scale),
        static_cast<int>(act_zp),
        bias_ptr != nullptr);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y.view(out_sizes);
}

}  // namespace dimq
