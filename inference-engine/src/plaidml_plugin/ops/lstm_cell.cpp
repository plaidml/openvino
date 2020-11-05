// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

edsl::Tensor clip_Tensor(bool should_clip, float clip, const edsl::Tensor& T) {
    if (should_clip) {
        return op::clip(T, edsl::Tensor(-clip), edsl::Tensor(clip));
    } else {
        return T;
    }
}

edsl::Tensor actication_func(std::string func_name, const edsl::Tensor& T) {
    if (func_name == "relu") {
        return op::relu(T);
    } else if (func_name == "sigmoid") {
        return op::sigmoid(T);
    } else if (func_name == "tanh") {
        return edsl::tanh(T);
    } else {
        THROW_IE_EXCEPTION << "Unsupported activation function";
    }
}

static OpRegistration reg("lstmcell", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 6);
    auto X = ctx.operands.at(0);  // input tensor
    auto H = ctx.operands.at(1);  // hidden state tensor
    auto C = ctx.operands.at(2);  // cell state tensor
    auto W = ctx.operands.at(3);  // weight tensor [4 * hidden_size, input_size]
    auto R = ctx.operands.at(4);  // recurrence weight tensor [4 * hidden_size, input_size]
    auto B = ctx.operands.at(5);  // bias tensor [4 * hidden_size]

    auto input_size = X.compute_shape().sizes().back();
    auto* layer = ngraph::as_type<ngraph::opset4::LSTMCell>(ctx.layer);
    auto hidden_size = layer->get_hidden_size();

    auto activations = layer->get_activations();
    auto activation_f = activations.at(0);
    auto activation_i = activations.at(1);
    auto activation_o = activations.at(2);

    auto activation_alpha = layer->get_activations_alpha();
    auto activation_beta = layer->get_activations_beta();

    auto clip = layer->get_clip();
    IE_ASSERT(clip > 0);
    auto should_clip = clip != std::numeric_limits<float>::infinity();

    auto W_f = op::slice(W).add_dim(0, hidden_size).add_dim(0, input_size);
    auto R_f = op::slice(R).add_dim(0, hidden_size).add_dim(0, hidden_size);
    auto B_f = op::slice(B).add_dim(0, hidden_size);
    auto T_f = op::dot(X, op::transpose(W_f)) + op::dot(H, op::transpose(R_f)) + B_f;
    auto T_clipped_f = clip_Tensor(should_clip, clip, T_f);
    auto f = actication_func(activation_f, T_clipped_f);

    auto W_i = op::slice(W).add_dim(hidden_size, 2 * hidden_size).add_dim(0, input_size);
    auto R_i = op::slice(R).add_dim(hidden_size, 2 * hidden_size).add_dim(0, hidden_size);
    auto B_i = op::slice(B).add_dim(hidden_size, 2 * hidden_size);
    auto T_i = op::dot(X, op::transpose(W_i)) + op::dot(H, op::transpose(R_i)) + B_i;
    auto T_clipped_i = clip_Tensor(should_clip, clip, T_i);
    auto i = actication_func(activation_i, T_clipped_i);

    auto W_c = op::slice(W).add_dim(2 * hidden_size, 3 * hidden_size).add_dim(0, input_size);
    auto R_c = op::slice(R).add_dim(2 * hidden_size, 3 * hidden_size).add_dim(0, hidden_size);
    auto B_c = op::slice(B).add_dim(2 * hidden_size, 3 * hidden_size);
    auto c = tanh(op::dot(X, op::transpose(W_c)) + op::dot(H, op::transpose(R_c)) + B_c);

    auto W_o = op::slice(W).add_dim(3 * hidden_size, 4 * hidden_size).add_dim(0, input_size);
    auto R_o = op::slice(R).add_dim(3 * hidden_size, 4 * hidden_size).add_dim(0, hidden_size);
    auto B_o = op::slice(B).add_dim(3 * hidden_size, 4 * hidden_size);
    auto T_o = op::dot(X, op::transpose(W_o)) + op::dot(H, op::transpose(R_o)) + B_o;
    auto T_clipped_o = clip_Tensor(should_clip, clip, T_o);
    auto o = actication_func(activation_o, T_clipped_o);

    auto C_o = f * C + i * c;
    auto H_o = o * tanh(C_o);

    return edsl::make_tuple(H_o, C_o);
});

}  // namespace PlaidMLPlugin
