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

namespace {

template <typename T>
std::vector<T> cast_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto* ngraph_const = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (ngraph_const) {
    return ngraph_const->cast_vector<T>();
  } else {
    THROW_IE_EXCEPTION << "Dynamic slicing not currently supported by PlaidML plugin; all of indices, offsets and default index"
                          "must be Constants.";
  }
}

}  // namespace

namespace PlaidMLPlugin {

static OpRegistration reg("EmbeddingBagOffsetsSum", [](const Context& ctx) {
  auto* layer = ngraph::as_type<ngraph::opset4::EmbeddingBagOffsetsSum>(ctx.layer);
  IE_ASSERT(ctx.operands.size() >= 3);
  IE_ASSERT(ctx.operands.size() <= 5);
  auto I = ctx.operands.at(0);
  auto indices = ctx.operands.at(1);
  IE_ASSERT(indices.rank() == 1);
  auto num_indices = indices.compute_shape().sizes()[0];
  auto offsets = cast_constant_operand<int32_t>(2, layer);
  int32_t default_index = -1;
  Tensor per_sample_weights;
  bool with_weights = false;

  if (ctx.operands.size() >= 4) {
    auto default_index_vec = cast_constant_operand<int32_t>(3, layer);
    default_index = default_index_vec[0];
    if (ctx.operands.size() == 5) {
      per_sample_weights = ctx.operands.at(4);
      with_weights = true;
    }
  }

  auto batch = offsets.size();
  offsets.push_back(num_indices);

  auto I_gathered = gather(I, indices);

  auto ndims = I_gathered.rank();
  std::vector<TensorDim> I_dims(ndims);
  std::vector<TensorIndex> I_idxs(ndims);
  std::vector<Tensor> slices, Os;
  I_gathered.bind_dims(I_dims);
  auto O_dims = I_dims;
  auto O_idxs = I_idxs;

  std::vector<TensorDim> w_dim(1);
  std::vector<TensorIndex> w_idx(1);
  auto wo_dim = w_dim;
  auto wo_idx = w_idx;
  if (with_weights == true) {
    per_sample_weights.bind_dims(w_dim);
  }

  wo_dim[0] = edsl::TensorDim(1);
  O_dims[0] = edsl::TensorDim(1);
  for (size_t i = 0; i < num_indices; ++i) {
    O_idxs[0] = I_idxs[0] - i;
    auto slice = edsl::Contraction(O_dims, O_idxs).sum(I_gathered(I_idxs)).build();
    if (with_weights == true) {
      wo_idx[0] = w_idx[0] - i;
      slice = slice * edsl::Contraction(wo_dim, wo_idx).sum(per_sample_weights(w_idx)).build();
    }
    slices.push_back(slice);
  }

  for (uint32_t l = 0; l < batch; ++l) {
    if (offsets[l + 1] == offsets[l]) {
      if (default_index == -1) {
        Os.push_back(slices[0] * 0);
      } else {
        O_idxs[0] = I_idxs[0] - default_index;
        Os.push_back(edsl::Contraction(O_dims, O_idxs).sum(I(I_idxs)));
      }
    } else {
      Tensor t = slices[offsets[l]];
      for (uint32_t i = offsets[l] + 1; i < offsets[l + 1]; ++i) {
        t = t + slices[i];
      }
      Os.push_back(t);
    }
  }

  auto O = op::concatenate(Os, 0);
  return edsl::make_tuple(O);
});

}  // namespace PlaidMLPlugin
