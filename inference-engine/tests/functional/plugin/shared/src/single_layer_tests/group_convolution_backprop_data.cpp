// Copyright (C) 2020 Intel Corporation
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

#include "single_layer_tests/group_convolution_backprop_data.hpp"

namespace LayerTestsDefinitions {

std::string GroupConvBackpropDataLayerTest::getTestCaseName(testing::TestParamInfo<groupConvBackpropDataLayerTestParamsSet> obj) {
    groupConvBackpropDataSpecificParams groupConvBackpropDataParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(groupConvBackpropDataParams, netPrecision, inputShapes, targetDevice) = obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels, numGroups;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvBackpropDataParams;

    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D_" << CommonTestUtils::vec2str(dilation) << "_";
    result << "O_" << convOutChannels << "_";
    result << "G_" << numGroups << "_";
    result << "AP_" << padType << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void GroupConvBackpropDataLayerTest::SetUp() {
    groupConvBackpropDataSpecificParams groupConvBackpropDataParams;
    std::vector<size_t> inputShape;
    std::tie(groupConvBackpropDataParams, netPrecision, inputShape, targetDevice) = this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels, numGroups;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvBackpropDataParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto groupConvBackpropData = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(
            ngraph::builder::makeGroupConvolutionBackpropData(paramOuts[0], ngPrc, kernel, stride, padBegin,
                                             padEnd, dilation, padType, convOutChannels, numGroups));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(groupConvBackpropData)};
    fnPtr = std::make_shared<ngraph::Function>(results, params, "GroupConvolutionBackpropData");
}

TEST_P(GroupConvBackpropDataLayerTest, CompareWithRefs) {
    inferAndValidate();
}
}  // namespace LayerTestsDefinitions
