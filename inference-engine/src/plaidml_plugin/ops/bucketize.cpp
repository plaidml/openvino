// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset3.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;  // NOLINT[build/namespaces]
using ngraph::opset3::Bucketize;

namespace PlaidMLPlugin {

static OpRegistration reg("Bucketize", [](const Context& ctx) {
  auto* layer = ngraph::as_type<Bucketize>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 2);
  auto A = ctx.operands.at(0);
  auto B = ctx.operands.at(1);

  std::vector<int> broadcastShape;
  std::vector<int> bcast_axes;
  auto A_shape = A.compute_shape().sizes();
  int count = A_shape.size();
  for (int i =0; i < count; i++) {
      // broadcast requires shape with int type
      broadcastShape.push_back(static_cast<int>(A_shape[i]));
      bcast_axes.push_back(i);
  }
  // Bucket is a 1-D tensor, the first dimension shall be tensor size
  auto B_shape = B.compute_shape().sizes();
  IE_ASSERT(B_shape.size() == 1);
  broadcastShape.push_back(static_cast<int>(B_shape[0]));
  auto broadcastResult = op::broadcast(A, broadcastShape, bcast_axes);

  auto outputType = to_plaidml(layer->get_output_type());
  auto one = edsl::cast(edsl::Tensor(1), outputType);
  auto zero = edsl::cast(edsl::Tensor(0), outputType);
  auto C = layer->get_with_right_bound() ?
      op::sum(select(broadcastResult > B, one, zero), edsl::make_tuple(count)) :
      op::sum(select(broadcastResult >= B, one, zero), edsl::make_tuple(count));
  return edsl::make_tuple(C);
});

}  // namespace PlaidMLPlugin
