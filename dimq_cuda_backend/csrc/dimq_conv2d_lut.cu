#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "dimq_common.cuh"

namespace dimq {

torch::Tensor build_product_table_cuda(torch::Tensor codebook, double act_scale, int64_t act_zp);

namespace {

template <typename scalar_t>
__global__ void dimq_conv2d_lut_kernel(const scalar_t* __restrict__ x,
                                       const uint8_t* __restrict__ packed_w,
                                       const float* __restrict__ product_table,
                                       const float* __restrict__ bias,
                                       float* __restrict__ y,
                                       int B,
                                       int C,
                                       int H,
                                       int W,
                                       int OC,
                                       int IC_PER_GROUP,
                                       int KH,
                                       int KW,
                                       int OH,
                                       int OW,
                                       int stride_h,
                                       int stride_w,
                                       int pad_h,
                                       int pad_w,
                                       int dilation_h,
                                       int dilation_w,
                                       int groups,
                                       float act_scale,
                                       int act_zp,
                                       bool has_bias) {
  __shared__ float table_s[256];
  int local = threadIdx.x;
  for (int i = local; i < 256; i += blockDim.x) {
    table_s[i] = product_table[i];
  }
  __syncthreads();

  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = static_cast<int64_t>(B) * OC * OH * OW;
  if (linear >= total) {
    return;
  }

  int ow = linear % OW;
  int oh = (linear / OW) % OH;
  int co = (linear / (OW * OH)) % OC;
  int b = linear / (static_cast<int64_t>(OW) * OH * OC);
  int oc_per_group = OC / groups;
  int group = co / oc_per_group;

  float acc = 0.0f;
  int64_t w_base = static_cast<int64_t>(co) * IC_PER_GROUP * KH * KW;
  for (int ci_g = 0; ci_g < IC_PER_GROUP; ++ci_g) {
    int ci = group * IC_PER_GROUP + ci_g;
    for (int kh = 0; kh < KH; ++kh) {
      int ih = oh * stride_h + kh * dilation_h - pad_h;
      if (ih < 0 || ih >= H) {
        continue;
      }
      for (int kw = 0; kw < KW; ++kw) {
        int iw = ow * stride_w + kw * dilation_w - pad_w;
        if (iw < 0 || iw >= W) {
          continue;
        }
        int64_t x_idx = ((static_cast<int64_t>(b) * C + ci) * H + ih) * W + iw;
        int64_t w_idx_linear = w_base + ((static_cast<int64_t>(ci_g) * KH + kh) * KW + kw);
        uint8_t a_idx = quantize_u4(scalar_to_float(x[x_idx]), act_scale, act_zp);
        uint8_t w_idx = get_u4(packed_w, w_idx_linear);
        acc += table_s[static_cast<int>(w_idx) * 16 + static_cast<int>(a_idx)];
      }
    }
  }
  if (has_bias) {
    acc += bias[co];
  }
  y[linear] = acc;
}

}  // namespace

torch::Tensor conv2d_lut_cuda(torch::Tensor x,
                              torch::Tensor packed_weight,
                              torch::Tensor codebook,
                              double act_scale,
                              int64_t act_zp,
                              c10::optional<torch::Tensor> bias,
                              int64_t out_channels,
                              int64_t in_channels_per_group,
                              int64_t kernel_h,
                              int64_t kernel_w,
                              int64_t stride_h,
                              int64_t stride_w,
                              int64_t pad_h,
                              int64_t pad_w,
                              int64_t dilation_h,
                              int64_t dilation_w,
                              int64_t groups) {
  TORCH_CHECK(x.is_cuda(), "conv2d_lut expects CUDA input");
  TORCH_CHECK(x.dim() == 4, "conv2d_lut expects NCHW input");
  TORCH_CHECK(packed_weight.is_cuda(), "conv2d_lut expects CUDA packed_weight");
  TORCH_CHECK(codebook.is_cuda(), "conv2d_lut expects CUDA codebook");
  TORCH_CHECK(x.is_floating_point(), "conv2d_lut expects floating input");
  TORCH_CHECK(packed_weight.scalar_type() == torch::kUInt8, "packed_weight must be uint8");
  TORCH_CHECK(codebook.is_floating_point(), "codebook must be floating point");
  TORCH_CHECK(codebook.numel() == 16, "codebook must have 16 entries for W4");
  TORCH_CHECK(act_scale > 0.0, "act_scale must be positive");
  TORCH_CHECK(act_zp >= 0 && act_zp <= 15, "act_zero_point must be in [0, 15]");
  TORCH_CHECK(out_channels > 0 && in_channels_per_group > 0, "channel counts must be positive");
  TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "kernel size must be positive");
  TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be positive");
  TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilation must be positive");
  TORCH_CHECK(groups > 0, "groups must be positive");

  int64_t B = x.size(0);
  int64_t C = x.size(1);
  int64_t H = x.size(2);
  int64_t W = x.size(3);
  TORCH_CHECK(C == in_channels_per_group * groups, "input channels do not match weight/groups");
  TORCH_CHECK(out_channels % groups == 0, "out_channels must be divisible by groups");
  int64_t out_h = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
  int64_t out_w = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
  TORCH_CHECK(out_h >= 0 && out_w >= 0, "computed negative output size");
  int64_t weight_numel = out_channels * in_channels_per_group * kernel_h * kernel_w;
  TORCH_CHECK(packed_weight.numel() >= (weight_numel + 1) / 2,
              "packed_weight is too small for requested conv weight shape");

  auto x_contig = x.contiguous();
  auto packed = packed_weight.contiguous().view({-1});
  auto table = build_product_table_cuda(codebook, act_scale, act_zp).contiguous().view({256});

  c10::optional<torch::Tensor> bias_float = c10::nullopt;
  const float* bias_ptr = nullptr;
  if (bias.has_value() && bias->defined()) {
    TORCH_CHECK(bias->is_cuda(), "bias must be CUDA when provided");
    TORCH_CHECK(bias->numel() == out_channels, "bias numel must equal out_channels");
    bias_float = bias->contiguous().to(torch::kFloat32).view({out_channels});
    bias_ptr = bias_float->data_ptr<float>();
  }

  auto y = torch::empty({B, out_channels, out_h, out_w}, x_contig.options().dtype(torch::kFloat32));
  int64_t total = y.numel();
  if (total == 0) {
    return y;
  }
  int threads = 256;
  int64_t blocks = (total + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x_contig.scalar_type(), "conv2d_lut_cuda", [&] {
    dimq_conv2d_lut_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x_contig.data_ptr<scalar_t>(),
        packed.data_ptr<uint8_t>(),
        table.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(out_channels),
        static_cast<int>(in_channels_per_group),
        static_cast<int>(kernel_h),
        static_cast<int>(kernel_w),
        static_cast<int>(out_h),
        static_cast<int>(out_w),
        static_cast<int>(stride_h),
        static_cast<int>(stride_w),
        static_cast<int>(pad_h),
        static_cast<int>(pad_w),
        static_cast<int>(dilation_h),
        static_cast<int>(dilation_w),
        static_cast<int>(groups),
        static_cast<float>(act_scale),
        static_cast<int>(act_zp),
        bias_ptr != nullptr);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

}  // namespace dimq
