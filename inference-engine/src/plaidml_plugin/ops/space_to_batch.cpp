// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset2.hpp"
#include "plaidml/op/op.h"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

static OpRegistration reg("SpaceToBatch", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 4);
    auto I = ctx.operands.at(0);
    auto block_shape = get_shape_from_constant_operand(1, ctx.layer);
    if (block_shape.size() != I.rank()) {
        THROW_IE_EXCEPTION << "block_shape and tensor dims have to equal!";
    }
    if (block_shape[0] != 1) {
        THROW_IE_EXCEPTION << "block_shape[0] is expected to be one.";
    }
    auto crops_begin = get_coords_from_constant_operand(2, ctx.layer);
    auto crops_end = get_coords_from_constant_operand(3, ctx.layer);

    // padding the original Tensor.
    std::vector<int> lo_pads;
    for (auto pad : crops_begin) {
        lo_pads.push_back(pad);
    }
    std::vector<int> hi_pads;
    for (auto pad : crops_end) {
        hi_pads.push_back(pad);
    }
    if (lo_pads[0] || hi_pads[0]) {
        THROW_IE_EXCEPTION << "batch dim pad have to be zero!";
    }
    edsl::Tensor padding_I = op::explicit_padding(I, lo_pads, hi_pads).padval(edsl::Constant(0));

    // reshape input tensor dim.
    std::vector<edsl::TensorDim> I_dims(padding_I.rank());
    padding_I.bind_dims(I_dims);

    std::vector<edsl::TensorDim> temp_dims;
    temp_dims.push_back(I_dims[0]);
    for (size_t i = 1; i < block_shape.size(); i++) {
        temp_dims.emplace_back(I_dims[i] / block_shape[i]);
        temp_dims.emplace_back(block_shape[i]);
    }
    auto reshape_I = edsl::reshape(padding_I, temp_dims);

    // transpose padding tensor
    std::vector<edsl::TensorDim> reshape_dims(reshape_I.rank());
    reshape_I.bind_dims(reshape_dims);
    auto dims_size = reshape_dims.size();
    // sort the dims.
    for (int i = dims_size - 2; i > 0; i = i - 2) {
        for (int j = 0; j < (dims_size - i) / 2; j++) {
            std::swap(reshape_dims[i + j], reshape_dims[i + j + 1]);
        }
    }
    for (size_t i = 0; i < dims_size / 2; i++) {
        std::swap(reshape_dims[i], reshape_dims[i + 1]);
    }
    std::vector<edsl::Value> dims_wrapper;
    for (auto dim : reshape_dims) {
        dims_wrapper.emplace_back(dim);
    }
    // TODO transpose got some problem.
    auto transpose_I = op::transpose(reshape_I, edsl::Value(dims_wrapper));

    // final reshape.
    size_t total_block_size = 1;
    for (size_t i = 0; i < block_shape.size(); i++) {
        total_block_size *= block_shape[i];
    }
    for (size_t i = 0; i < I_dims.size(); i++) {
        if (i == 0) {
            I_dims[i] = I_dims[i] * total_block_size;
        }
        I_dims[i] = I_dims[i] / block_shape[i];
    }
    edsl::Tensor O = edsl::reshape(transpose_I, I_dims);

    return edsl::make_tuple(O);
});

}  // namespace PlaidMLPlugin
