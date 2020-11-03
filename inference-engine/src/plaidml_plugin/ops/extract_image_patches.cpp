// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;         // NOLINT[build/namespaces]
using namespace InferenceEngine; // NOLINT[build/namespaces]

namespace {

template <typename T>
edsl::Tensor createKernelTensor(std::vector<int64_t> &filterShape,
                                edsl::Tensor inputTensor) {
  auto inputType = inputTensor.dtype();
  auto inputShape = inputTensor.compute_shape().sizes();
  auto inputChannel = inputShape[1];
  // build kernel Tensor dims.
  auto depths = filterShape[0] * filterShape[1] * inputChannel;
  std::vector<int64_t> kernelDims{
      /*output channels*/ depths,
      /*input channels*/ inputChannel,
      /*filter width*/ filterShape[0],
      /*filter height*/ filterShape[1],
  };
  // kernel tensor element size.
  size_t kernelSum = 1;
  for (auto dim : kernelDims) {
    kernelSum *= dim;
  }
  // build one-zero kernel Tensor.
  std::vector<T> data(kernelSum, 0);

  int64_t channel = 0;
  for (int64_t depth = 0; depth < depths; depth++) {
    auto index = depth * kernelDims[1] * kernelDims[2] * kernelDims[3] +
                 channel * kernelDims[2] * kernelDims[3] +
                 (depth / inputChannel) / filterShape[1] * kernelDims[3] +
                 (depth / inputChannel) % filterShape[1];
    data[index] = 1;
    if (++channel == inputChannel) {
      channel = 0;
    }
  }
  TensorShape shape(inputType, kernelDims);
  Buffer buffer(shape);
  buffer.copy_from(data.data());

  return edsl::cast(edsl::Constant(buffer, "Kernel"), inputType);
}

} // namespace

namespace PlaidMLPlugin {

static OpRegistration reg("ExtractImagePatches", [](const Context &ctx) {
  auto *layer = ngraph::as_type<ngraph::opset3::ExtractImagePatches>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 1);

  auto inputTensor = ctx.operands.at(0);
  std::vector<int64_t> filterShape;
  for (auto dim : layer->get_sizes()) {
    filterShape.push_back(dim);
  }

  edsl::Tensor kernelTensor;
  switch (inputTensor.dtype()) {
  case DType::FLOAT32:
    kernelTensor = createKernelTensor<float>(filterShape, inputTensor);
    break;
  case DType::INT32:
    kernelTensor = createKernelTensor<int>(filterShape, inputTensor);
    break;
  }

  std::vector<size_t> strides;
  for (auto stride : layer->get_strides()) {
    strides.push_back(stride);
  }

  std::vector<size_t> dilations;
  for (auto dilation : layer->get_rates()) {
    dilations.push_back(dilation);
  }

  auto autopad_mode = to_plaidml(layer->get_auto_pad());
  if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
    THROW_IE_EXCEPTION << "only valid or auto_pad(same_upper or same_lower) "
                          "PadType is accepted";
  }

  auto result = op::convolution(inputTensor, kernelTensor)
                    .strides(strides)
                    .dilations(dilations)
                    .autopad_mode(autopad_mode)
                    .input_layout(plaidml::op::TensorLayout::NCX)
                    .filter_layout(plaidml::op::TensorLayout::KCX);
  return edsl::make_tuple(result);
});

} // namespace PlaidMLPlugin
