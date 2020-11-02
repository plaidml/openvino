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

namespace PlaidMLPlugin {

static OpRegistration reg("ExtractImagePatches", [](const Context &ctx) {
  auto *layer = ngraph::as_type<ngraph::opset3::ExtractImagePatches>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 1);

  auto inputTensor = ctx.operands.at(0);
  auto inputType = inputTensor.dtype();
  auto inputShape = inputTensor.compute_shape().sizes();

  std::vector<int64_t> filterShape;
  for (auto dim : layer->get_sizes()) {
    filterShape.push_back(dim);
  }
  // build kernel Tensor dims.
  auto depths = filterShape[0] * filterShape[1];
  std::vector<int64_t> kernelDims{
      /*output channels*/ depths,
      /*input channels*/ inputShape[1],
      /*filter width*/ filterShape[0],
      /*filter height*/ filterShape[1],
  };
  // kernel tensor element size.
  size_t kernelSum = 1;
  for (auto dim : kernelDims) {
    kernelSum *= dim;
  }
  // build one-zero kernel Tensor.
  // TODO get element type from DTYPE.
  std::vector<int> data(kernelSum, 0);
  for (int64_t depth = 0; depth < depths; depth++) {
    for (int64_t channel = 0; channel < inputShape[1]; channel++) {
      auto index = channel * kernelDims[1] * kernelDims[2] * kernelDims[3] +
                   depth * kernelDims[2] * kernelDims[3] +
                   depth / filterShape[0] * kernelDims[3] +
                   depth % filterShape[0];
      data[index] = 1;
    }
  }
  TensorShape shape(inputType, kernelDims);
  Buffer buffer(shape);
  buffer.copy_from(data.data());

  auto kernelTensor = edsl::cast(edsl::Constant(buffer, "Kernel"), inputType);

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
