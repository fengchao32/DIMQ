#include <torch/extension.h>

namespace dimq {

torch::Tensor pack_u4_cuda(torch::Tensor indices);
torch::Tensor unpack_u4_cuda(torch::Tensor packed, int64_t numel);
torch::Tensor quantize_activation_u4_cuda(torch::Tensor x, double act_scale, int64_t act_zp);
torch::Tensor build_product_table_cuda(torch::Tensor codebook, double act_scale, int64_t act_zp);
torch::Tensor linear_lut_cuda(torch::Tensor x,
                              torch::Tensor packed_weight,
                              torch::Tensor codebook,
                              double act_scale,
                              int64_t act_zp,
                              c10::optional<torch::Tensor> bias,
                              int64_t out_features,
                              int64_t in_features);
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
                              int64_t groups);

}  // namespace dimq

TORCH_LIBRARY(dimq, m) {
  m.def("pack_u4(Tensor indices) -> Tensor");
  m.def("unpack_u4(Tensor packed, int numel) -> Tensor");
  m.def("quantize_activation_u4(Tensor x, float act_scale, int act_zp) -> Tensor");
  m.def("build_product_table(Tensor codebook, float act_scale, int act_zp) -> Tensor");
  m.def("linear_lut(Tensor x, Tensor packed_w, Tensor codebook, float act_scale, int act_zp, Tensor? bias, int out_features, int in_features) -> Tensor");
  m.def("conv2d_lut(Tensor x, Tensor packed_w, Tensor codebook, float act_scale, int act_zp, Tensor? bias, int out_channels, int in_channels_per_group, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, int dilation_h, int dilation_w, int groups) -> Tensor");
}

TORCH_LIBRARY_IMPL(dimq, CUDA, m) {
  m.impl("pack_u4", &dimq::pack_u4_cuda);
  m.impl("unpack_u4", &dimq::unpack_u4_cuda);
  m.impl("quantize_activation_u4", &dimq::quantize_activation_u4_cuda);
  m.impl("build_product_table", &dimq::build_product_table_cuda);
  m.impl("linear_lut", &dimq::linear_lut_cuda);
  m.impl("conv2d_lut", &dimq::conv2d_lut_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_u4", &dimq::pack_u4_cuda, "Pack uint8 indices into u4 bytes");
  m.def("unpack_u4", &dimq::unpack_u4_cuda, "Unpack u4 bytes into uint8 indices");
  m.def("quantize_activation_u4", &dimq::quantize_activation_u4_cuda, "Uniform affine u4 activation quantization");
  m.def("build_product_table", &dimq::build_product_table_cuda, "Build DIMQ W4A4 product table");
  m.def("linear_lut", &dimq::linear_lut_cuda, "DIMQ packed-index Linear LUT forward");
  m.def("conv2d_lut", &dimq::conv2d_lut_cuda, "DIMQ packed-index Conv2d LUT forward");
}
