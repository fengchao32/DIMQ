// Reserved for a future CUTLASS/CuTe tiled implementation.
//
// The current backend exposes correctness-first CUDA SIMT LUT kernels in
// dimq_gemm_lut.cu and dimq_conv2d_lut.cu.  Those kernels intentionally do not
// treat non-uniform codebook indices as affine INT4 values.  A future optimized
// kernel can keep the same Python/operator ABI and replace the accumulation
// loop with CUTLASS-style tile iterators plus shared-memory product-table loads.
