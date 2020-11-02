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
static edsl::Tensor buildFlatFilter(const std::vector<size_t> &shape){

}

static OpRegistration reg("ExtractImagePatches", [](const Context &ctx) {
  auto *layer = ngraph::as_type<ngraph::opset3::ExtractImagePatches>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 1);

  auto inputTensor = ctx.operands.at(0);
  std::vector<size_t> filterShape;
  for (auto dim : layer->get_sizes()){
    filterShape.push_back(dim);
  }
  auto filterTensor = buildFlatFilter(filterShape);

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

  auto result = op::convolution(inputTensor, filterTensor)
                    .strides(strides)
                    .dilations(dilations)
                    .autopad_mode(autopad_mode)
                    .input_layout(plaidml::op::TensorLayout::NCX)
                    .filter_layout(plaidml::op::TensorLayout::KCX);
  return edsl::make_tuple(result);
});

} // namespace PlaidMLPlugin
