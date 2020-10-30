// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;         // NOLINT[build/namespaces]
using namespace InferenceEngine; // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("binaryconvolution", [](const Context &ctx) {
  auto *layer = ngraph::as_type<ngraph::opset4::BinaryConvolution>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 2);
  auto I = ctx.operands.at(0);
  auto F = ctx.operands.at(1);
  std::vector<int> strides;
  for (auto stride : layer->get_strides()) {
    strides.push_back(stride);
  }
  std::vector<int> dilations;
  for (auto dilation : layer->get_dilations()) {
    dilations.push_back(dilation);
  }

  auto mode = layer->get_mode();
  edsl::Tensor inputTensor, filterTensor;
  if (mode ==
      ngraph::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT) {
    auto one = cast(edsl::Tensor(1.0), DType::FLOAT32);
    auto minusOne = cast(edsl::Tensor(-1.0), DType::FLOAT32);
    inputTensor = edsl::select(I == 0, minusOne, one);
    filterTensor = edsl::select(F == 0, minusOne, one);
  } else {
    auto one = cast(edsl::Tensor(1.0), DType::FLOAT32);
    auto zero = cast(edsl::Tensor(0.0), DType::FLOAT32);
    inputTensor = edsl::select(I == 0, zero, one);
    filterTensor = edsl::select(F == 0, zero, one);
  }

  auto autopad_mode = to_plaidml(layer->get_auto_pad());
  auto pad_value = layer->get_pad_value();
  auto result = op::convolution(inputTensor, filterTensor)
                    .strides(strides)
                    .dilations(dilations)
                    .autopad_mode(autopad_mode)
                    .input_layout(plaidml::op::TensorLayout::NCX)
                    .filter_layout(plaidml::op::TensorLayout::KCX);
  if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
    // WARNING: float pad value is passes here
    int padding = pad_value;
    result.manual_padding({padding});
  }
  return edsl::make_tuple(result);
});

} // namespace PlaidMLPlugin