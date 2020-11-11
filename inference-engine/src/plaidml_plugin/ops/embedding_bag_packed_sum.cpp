// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using namespace plaidml::edsl;

namespace PlaidMLPlugin {

static OpRegistration reg("EmbeddingBagPackedSum", [](const Context& ctx) {
  auto* layer = ngraph::as_type<ngraph::opset4::EmbeddingBagPackedSum>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 2 || ctx.operands.size() == 3);
  auto I = ctx.operands.at(0);
  auto indices = ctx.operands.at(1);
  IE_ASSERT(indices.rank() == 2);
  auto batch = indices.compute_shape().sizes()[0];
  auto indices_per_bag = indices.compute_shape().sizes()[1];

  Tensor per_sample_weights;
  bool with_weights = false;

  if (ctx.operands.size() == 3) {
    per_sample_weights = ctx.operands.at(2);
    IE_ASSERT(per_sample_weights.rank() == 2);
    with_weights = true;
  }

  std::vector<Tensor> Os;

  for (uint32_t i = 0; i < batch; ++i) {
    Tensor i_slice = op::slice(indices).add_dim(i, i + 1).add_dim(0, indices_per_bag);
    i_slice = op::reshape(i_slice, edsl::make_tuple<size_t>(indices_per_bag));
    Tensor w_slice;
    if (with_weights == true) {
      w_slice = op::slice(per_sample_weights).add_dim(i, i + 1).add_dim(0, indices_per_bag);
      w_slice = op::reshape(w_slice, edsl::make_tuple<size_t>(indices_per_bag));
    }
    auto I_gathered = gather(I, i_slice);
    auto ndims = I_gathered.rank();
    std::vector<TensorDim> I_dims(ndims);
    std::vector<TensorIndex> I_idxs(ndims);
    I_gathered.bind_dims(I_dims);
    auto O_dims = I_dims;
    auto O_idxs = I_idxs;

    Tensor sum;
    O_dims[0] = edsl::TensorDim(1);
    for (uint32_t j = 0; j < indices_per_bag; ++j) {
      O_idxs[0] = I_idxs[0] - j;
      auto slice = edsl::Contraction(O_dims, O_idxs).assign(I_gathered(I_idxs)).build();
      if (with_weights == true) {
        Tensor weight = op::slice(w_slice).add_dim(j, j+1);
        slice = slice * weight;
      }
      if (j == 0) {
        sum = slice;
      } else {
        sum = sum + slice;
      }
    }

    Os.push_back(sum);
  }

  auto O = op::concatenate(Os, 0);
  return edsl::make_tuple(O);
});

}  // namespace PlaidMLPlugin
