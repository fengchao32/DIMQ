#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "dimq_common.cuh"

namespace dimq {
namespace {

__global__ void pack_u4_kernel(const uint8_t* __restrict__ indices,
                               uint8_t* __restrict__ packed,
                               int64_t numel) {
  int64_t byte_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t base = byte_idx * 2;
  if (base >= numel) {
    return;
  }
  uint8_t low = indices[base] & 0x0F;
  uint8_t high = 0;
  if (base + 1 < numel) {
    high = (indices[base + 1] & 0x0F) << 4;
  }
  packed[byte_idx] = low | high;
}

__global__ void unpack_u4_kernel(const uint8_t* __restrict__ packed,
                                 uint8_t* __restrict__ indices,
                                 int64_t numel) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  indices[idx] = get_u4(packed, idx);
}

template <typename scalar_t>
__global__ void quantize_activation_u4_kernel(const scalar_t* __restrict__ x,
                                              uint8_t* __restrict__ out,
                                              int64_t numel,
                                              float scale,
                                              int zero_point) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numel) {
    return;
  }
  out[idx] = quantize_u4(scalar_to_float(x[idx]), scale, zero_point);
}

}  // namespace

torch::Tensor pack_u4_cuda(torch::Tensor indices) {
  TORCH_CHECK(indices.is_cuda(), "pack_u4 expects a CUDA tensor");
  TORCH_CHECK(indices.scalar_type() == torch::kUInt8, "pack_u4 expects uint8 indices");
  auto contiguous = indices.contiguous().view({-1});
  int64_t numel = contiguous.numel();
  auto packed = torch::empty({(numel + 1) / 2}, contiguous.options());
  if (numel == 0) {
    return packed;
  }
  int threads = 256;
  int64_t blocks = (packed.numel() + threads - 1) / threads;
  pack_u4_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      contiguous.data_ptr<uint8_t>(), packed.data_ptr<uint8_t>(), numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return packed;
}

torch::Tensor unpack_u4_cuda(torch::Tensor packed, int64_t numel) {
  TORCH_CHECK(packed.is_cuda(), "unpack_u4 expects a CUDA tensor");
  TORCH_CHECK(packed.scalar_type() == torch::kUInt8, "unpack_u4 expects uint8 packed data");
  TORCH_CHECK(numel >= 0, "numel must be non-negative");
  TORCH_CHECK(packed.numel() >= (numel + 1) / 2, "packed tensor is too small for requested numel");
  auto contiguous = packed.contiguous().view({-1});
  auto indices = torch::empty({numel}, contiguous.options());
  if (numel == 0) {
    return indices;
  }
  int threads = 256;
  int64_t blocks = (numel + threads - 1) / threads;
  unpack_u4_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      contiguous.data_ptr<uint8_t>(), indices.data_ptr<uint8_t>(), numel);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return indices;
}

torch::Tensor quantize_activation_u4_cuda(torch::Tensor x, double act_scale, int64_t act_zp) {
  TORCH_CHECK(x.is_cuda(), "quantize_activation_u4 expects a CUDA tensor");
  TORCH_CHECK(x.is_floating_point(), "quantize_activation_u4 expects floating input");
  TORCH_CHECK(act_scale > 0.0, "act_scale must be positive");
  TORCH_CHECK(act_zp >= 0 && act_zp <= 15, "act_zero_point must be in [0, 15]");
  auto contiguous = x.contiguous();
  auto out = torch::empty(contiguous.sizes(), contiguous.options().dtype(torch::kUInt8));
  if (contiguous.numel() == 0) {
    return out;
  }
  int threads = 256;
  int64_t blocks = (contiguous.numel() + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(contiguous.scalar_type(), "quantize_activation_u4_cuda", [&] {
    quantize_activation_u4_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        contiguous.data_ptr<scalar_t>(),
        out.data_ptr<uint8_t>(),
        contiguous.numel(),
        static_cast<float>(act_scale),
        static_cast<int>(act_zp));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace dimq
