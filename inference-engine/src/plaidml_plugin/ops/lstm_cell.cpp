// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

/*
Formula in OpenVINO Doc is not very accurate, it should be:
   *  - matrix mult
  (.) - eltwise mult
  [,] - concatenation
  sigm - 1/(1 + e^{-x})
  tanh - (e^{2x} - 1)/(e^{2x} + 1)
  f = sigm(X*Wf^T + Hi*Rf^T + Bf)
  i = sigm(X*Wi^T + Hi*Ri^T + Bi)
  c = tanh(X*Wc^T + Hi*Rc^T + Bc)
  o = sigm(X*Wo^T + Hi*Ro^T+ Bo)
  Co = f (.) Ci + i (.) c
  Ho = o (.) tanh(Co)
*/

void print_shape(edsl::Tensor t) {
  auto shape = t.compute_shape();
  for (auto i : shape.sizes()) {
    std::cout << i << "  ";
  }
  std::cout << std::endl;
}

static OpRegistration reg("lstmcell", [](const Context &ctx) {
  IE_ASSERT(ctx.operands.size() == 6);
  auto X = ctx.operands.at(0);  // input tensor
  auto H = ctx.operands.at(1);  // hidden state tensor
  auto C = ctx.operands.at(2);  // cell state tensor
  auto W = ctx.operands.at(3);  // weight tensor [4 * hidden_size, input_size]
  auto R = ctx.operands.at(
      4);  // recurrence weight tensor [4 * hidden_size, input_size]
  auto B = ctx.operands.at(5);  // bias tensor [4 * hidden_size]

  auto *layer = ngraph::as_type<ngraph::opset4::LSTMCell>(ctx.layer);
  auto input_size = X.compute_shape().sizes().back();
  auto hidden_size = layer->get_hidden_size();
  // TODO: apply optional activation and value clip
  // auto activations = layer->get_activations();
  // auto activation_alpha = layer->get_activations_alpha();
  // auto activation_beta = layer->get_activations_beta();
  // auto clip = layer->get_clip();

  auto W_f = op::slice(W).add_dim(0, hidden_size).add_dim(0, input_size);
  auto R_f = op::slice(R).add_dim(0, hidden_size).add_dim(0, hidden_size);
  auto B_f = op::slice(B).add_dim(0, hidden_size);
  auto f = op::sigmoid(op::dot(X, op::transpose(W_f)) + op::dot(H, op::transpose(R_f)) + B_f);

  auto W_i = op::slice(W).add_dim(hidden_size, 2 * hidden_size).add_dim(0, input_size);
  auto R_i = op::slice(R)
                 .add_dim(hidden_size, 2 * hidden_size)
                 .add_dim(0, hidden_size);
  auto B_i = op::slice(B).add_dim(hidden_size, 2 * hidden_size);
  auto i = op::sigmoid(op::dot(X, op::transpose(W_i)) + op::dot(H, op::transpose(R_i)) + B_i);

  auto W_c = op::slice(W)
                 .add_dim(2 * hidden_size, 3 * hidden_size)
                 .add_dim(0, input_size);
  auto R_c = op::slice(R)
                 .add_dim(2 * hidden_size, 3 * hidden_size)
                 .add_dim(0, hidden_size);
  auto B_c = op::slice(B).add_dim(2 * hidden_size, 3 * hidden_size);
  auto c = op::sigmoid(op::dot(X, op::transpose(W_c)) + op::dot(H, op::transpose(R_c)) + B_c);

  auto W_o = op::slice(W)
                 .add_dim(3 * hidden_size, 4 * hidden_size)
                 .add_dim(0, input_size);
  auto R_o = op::slice(R)
                 .add_dim(3 * hidden_size, 4 * hidden_size)
                 .add_dim(0, hidden_size);
  auto B_o = op::slice(B).add_dim(3 * hidden_size, 4 * hidden_size);
  auto o = op::sigmoid(op::dot(X, op::transpose(W_o)) + op::dot(H, op::transpose(R_o)) + B_o);

  auto C_o = f * C + i * c;
  auto H_o = o * tanh(C_o);

  return edsl::make_tuple(H_o, C_o);
});

}  // namespace PlaidMLPlugin
