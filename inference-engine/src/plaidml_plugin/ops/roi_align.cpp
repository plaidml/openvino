// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset3.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;         // NOLINT[build/namespaces]
using namespace InferenceEngine; // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("roialign", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 3);
    auto X = ctx.operands.at(0);
    auto roi_boxes = ctx.operands.at(1);
    auto batch_indices = ctx.operands.at(2);

    auto *layer = ngraph::as_type<ngraph::opset3::ROIAlign>(ctx.layer);
    auto pooled_h = layer->get_pooled_h();
    auto pooled_w = layer->get_pooled_w();
    auto sampling_ratio = layer->get_sampling_ratio();
    auto spatial_scale = layer->get_spatial_scale();
    auto mode = layer->get_mode();

    auto input_size = X.compute_shape().sizes();
    edsl::TensorIndex i, j, k;

    return edsl::make_tuple();
}