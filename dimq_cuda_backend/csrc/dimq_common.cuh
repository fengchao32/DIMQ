#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace dimq {

__device__ __forceinline__ uint8_t get_u4(const uint8_t* packed, int64_t idx) {
  uint8_t byte = packed[idx >> 1];
  if ((idx & 1) == 0) {
    return byte & 0x0F;
  }
  return (byte >> 4) & 0x0F;
}

__device__ __forceinline__ uint8_t quantize_u4(float x, float scale, int zero_point) {
  int q = __float2int_rn(x / scale) + zero_point;
  q = max(0, min(15, q));
  return static_cast<uint8_t>(q);
}

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t value) {
  return static_cast<float>(value);
}

}  // namespace dimq
