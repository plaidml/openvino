// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>

#include "ngraph_functions/utils/data_utils.hpp"

namespace ngraph {
namespace builder {

ngraph::ParameterVector makeParams(const element::Type &type, const std::vector<std::vector<size_t>> &shapes);

std::shared_ptr<ngraph::Node> makeConstant(const element::Type &type, const std::vector<size_t> &shape,
                                           const std::vector<float> &data, bool random = false);

std::shared_ptr<ngraph::Node> makeConvolution(const ngraph::Output<Node> &in,
                                              const element::Type &type,
                                              const std::vector<size_t> &filterSize,
                                              const std::vector<size_t> &strides,
                                              const std::vector<ptrdiff_t> &padsBegin,
                                              const std::vector<ptrdiff_t> &padsEnd,
                                              const std::vector<size_t> &dilations,
                                              const op::PadType &autoPad,
                                              size_t numOutChannels,
                                              bool addBiases = false,
                                              const std::vector<float> &filterWeights = {},
                                              const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                          const element::Type &type,
                                                          const std::vector<size_t> &filterSize,
                                                          const std::vector<size_t> &strides,
                                                          const std::vector<ptrdiff_t> &padsBegin,
                                                          const std::vector<ptrdiff_t> &padsEnd,
                                                          const std::vector<size_t> &dilations,
                                                          const op::PadType &autoPad,
                                                          size_t numOutChannels,
                                                          bool addBiases = false,
                                                          const std::vector<float> &filterWeights = {},
                                                          const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                          const ngraph::Output<Node> &weights,
                                                          const element::Type &type,
                                                          const std::vector<size_t> &strides,
                                                          const std::vector<ptrdiff_t> &padsBegin,
                                                          const std::vector<ptrdiff_t> &padsEnd,
                                                          const std::vector<size_t> &dilations,
                                                          const op::PadType &autoPad,
                                                          bool addBiases = false,
                                                          const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeGroupConvolution(const ngraph::Output<ngraph::Node> &in,
                                                   const element::Type &type,
                                                   const std::vector<size_t> &filterSize,
                                                   const std::vector<size_t> &strides,
                                                   const std::vector<ptrdiff_t> &padsBegin,
                                                   const std::vector<ptrdiff_t> &padsEnd,
                                                   const std::vector<size_t> &dilations,
                                                   const op::PadType &autoPad,
                                                   size_t numOutChannels,
                                                   size_t numGroups,
                                                   bool addBiases = false,
                                                   const std::vector<float> &filterWeights = {},
                                                   const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeGroupConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                               const element::Type &type,
                                                               const std::vector<size_t> &filterSize,
                                                               const std::vector<size_t> &strides,
                                                               const std::vector<ptrdiff_t> &padsBegin,
                                                               const std::vector<ptrdiff_t> &padsEnd,
                                                               const std::vector<size_t> &dilations,
                                                               const op::PadType &autoPad,
                                                               size_t numOutChannels,
                                                               size_t numGroups,
                                                               bool addBiases = false,
                                                               const std::vector<float> &filterWeights = {},
                                                               const std::vector<float> &biasesWeights = {});

std::shared_ptr<ngraph::Node> makeGroupConvolutionBackpropData(const ngraph::Output<Node> &in,
                                                               const ngraph::Output<Node> &weights,
                                                               const element::Type &type,
                                                               const std::vector<size_t> &strides,
                                                               const std::vector<ptrdiff_t> &padsBegin,
                                                               const std::vector<ptrdiff_t> &padsEnd,
                                                               const std::vector<size_t> &dilations,
                                                               const op::PadType &autoPad,
                                                               bool addBiases = false,
                                                               const std::vector<float> &biasesWeights = {});

std::shared_ptr<Node> makeFakeQuantize(const ngraph::Output<Node> &in,
                                       const element::Type &type,
                                       std::size_t levels,
                                       std::vector<size_t> constShapes);

std::shared_ptr<ngraph::Node> makeMVN(const ngraph::Output<Node> &in,
                                      bool acrossChannels,
                                      bool normalizeVariance,
                                      double eps);

std::shared_ptr<ngraph::Node> makeSplit(const ngraph::Output<Node> &in,
                                        const element::Type &type,
                                        size_t numSplits,
                                        size_t axis);

std::shared_ptr<ngraph::Node> makeActivation(const ngraph::Output<Node> &in,
                                             const element::Type &type,
                                             ngraph::helpers::ActivationTypes activationType);

std::shared_ptr<ngraph::Node> makeMatMul(const ngraph::Output<Node>& A,
                                         const ngraph::Output<Node>& B);

std::shared_ptr<ngraph::Node> makeSelect(std::vector<ngraph::Output<Node>> &in,
                                         const ngraph::op::AutoBroadcastSpec& auto_broadcast);

std::shared_ptr<ngraph::Node> makeBatchToSpace(const ngraph::Output<Node> &in,
                                               const element::Type &type,
                                               const std::vector<size_t> &blockShape,
                                               const std::vector<size_t> &cropsBegin,
                                               const std::vector<size_t> &cropsEnd);

std::shared_ptr<ngraph::Node> makeDepthToSpace(const ngraph::Output<Node> &in,
                                               ngraph::opset1::DepthToSpace::DepthToSpaceMode mode,
                                               size_t blockSize);

std::shared_ptr<ngraph::Node> makeSpaceToBatch(const ngraph::Output<Node> &in,
                                               const element::Type &type,
                                               const std::vector<size_t> &blockShape,
                                               const std::vector<size_t> &padsBegin,
                                               const std::vector<size_t> &padsEnd);

std::shared_ptr<ngraph::Node> makeSpaceToDepth(const ngraph::Output<Node> &in,
                                               ngraph::opset1::SpaceToDepth::SpaceToDepthMode mode,
                                               size_t blockSize);

std::shared_ptr<ngraph::Node> makeStridedSlice(const ngraph::Output<Node> &in,
                                               const std::vector<int64_t> &begin,
                                               const std::vector<int64_t> &end,
                                               const std::vector<int64_t> &stride,
                                               const element::Type &type,
                                               const std::vector<int64_t> &begin_mask,
                                               const std::vector<int64_t> &end_mask,
                                               const std::vector<int64_t> &new_axis_mask = std::vector<int64_t>{},
                                               const std::vector<int64_t> &shrink_mask = std::vector<int64_t>{},
                                               const std::vector<int64_t> &ellipsis_mask = std::vector<int64_t>{});

}  // namespace builder
}  // namespace ngraph
