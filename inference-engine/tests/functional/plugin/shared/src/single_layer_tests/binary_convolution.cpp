// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/binary_convolution.hpp"

namespace LayerTestsDefinitions {

std::string BinaryConvolutionLayerTest::getTestCaseName(testing::TestParamInfo<binConvLayerTestParamsSet> obj) {
  binConvSpecificParams binConvParams;
  InferenceEngine::Precision netPrecision;
  InferenceEngine::SizeVector inputShapes;
  std::string targetDevice;
  std::tie(binConvParams, netPrecision, inputShapes, targetDevice) = obj.param;
  ngraph::op::PadType padType;
  InferenceEngine::SizeVector kernel, stride, dilation;
  std::vector<ptrdiff_t> padBegin, padEnd;
  size_t binConvOutChannels;
  std::tie(kernel, stride, padBegin, padEnd, dilation, binConvOutChannels, padType) = binConvParams;

  std::ostringstream result;
  result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
  result << "K" << CommonTestUtils::vec2str(kernel) << "_";
  result << "S" << CommonTestUtils::vec2str(stride) << "_";
  result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
  result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
  result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
  result << "O=" << binConvOutChannels << "_";
  result << "AP=" << padType << "_";
  result << "netPRC=" << netPrecision.name() << "_";
  result << "targetDevice=" << targetDevice;
  return result.str();
}

void ConvolutionLayerTest::SetUp() {
  binConvSpecificParams binConvParams;
  std::vector<size_t> inputShape;
  auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
  std::tie(binConvParams, netPrecision, inputShape, targetDevice) = this->GetParam();
  ngraph::op::PadType padType;
  InferenceEngine::SizeVector kernel, stride, dilation;
  std::vector<ptrdiff_t> padBegin, padEnd;
  size_t binConvOutChannels;
  std::tie(kernel, stride, padBegin, padEnd, dilation, binConvOutChannels, padType) = binConvParams;
  auto ngPrc = FuncTestUtils::PrecisionUtils::binConvertIE2nGraphPrc(netPrecision);
  auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
  auto paramOuts = ngraph::helpers::binConvert2OutputVector(
      ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
  auto binConv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(
      ngraph::builder::makeConvolution(paramOuts[0], ngPrc, kernel, stride, padBegin,
                                       padEnd, dilation, padType, binConvOutChannels));
  ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(binConv)};
  function = std::make_shared<ngraph::Function>(results, params, "convolution");
}

TEST_P(BinaryConvolutionLayerTest, CompareWithRefs) {
Run();
}
}  // namespace LayerTestsDefinitions
