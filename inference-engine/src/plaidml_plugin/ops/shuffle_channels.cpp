// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("shuffleChannels", [](const Context& ctx) {
  auto* layer = ngraph::as_type<ngraph::opset3::ShuffleChannels>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);
  auto group = layer->get_group();
  //  the axis parameter doesn't show in the formula.
  //  not sure why we need axis, so just leave it alone for now.

  std::vector<edsl::TensorDim> original_dims(I.rank());
  I.bind_dims(original_dims);
  // the channel dim size have to be dividable by group.
  auto channel_dim = original_dims[1];

  // follow the openvino op spec
  // x' = reshape(x, [N, group, C / group, H * W])
  std::vector<edsl::TensorDim> channel_group_dims(I.rank());
  channel_group_dims[0] = original_dims[0];
  channel_group_dims[1] = edsl::TensorDim(group);
  channel_group_dims[2] = channel_dim / group;
  channel_group_dims[3] = original_dims[2] * original_dims[3];
  auto reshape_I = edsl::reshape(I, channel_group_dims);

  // x'' = transpose(x', [0, 2, 1, 3])
  std::vector<edsl::Value> dims_wrapper = {edsl::Value(0), edsl::Value(2), edsl::Value(1), edsl::Value(3)};
  auto transpose_I = op::transpose(reshape_I, edsl::Value(dims_wrapper));

  // y = reshape(x'', [N, C, H, W])
  auto O = edsl::reshape(transpose_I, original_dims);
  return edsl::make_tuple(O);
});

}  // namespace PlaidMLPlugin
