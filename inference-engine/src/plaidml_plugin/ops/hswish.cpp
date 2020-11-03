// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("hswish", [](const Context& ctx) {
  IE_ASSERT(ctx.operands.size() == 1);
  auto I = ctx.operands.at(0);

  auto ndims = I.rank();
  std::vector<edsl::TensorDim> I_dims(ndims);
  I.bind_dims(I_dims);

  // f(x) =  x * min(max(x + 3, 0), 6) / 6
  // f(x) = x * min(ReLU(x + 3), 6) / 6
  auto R = op::minimum(op::relu(I+3), edsl::cast(edsl::Tensor(6), I.dtype()));
  return edsl::make_tuple(I * R / 6);
});

}  // namespace PlaidMLPlugin
