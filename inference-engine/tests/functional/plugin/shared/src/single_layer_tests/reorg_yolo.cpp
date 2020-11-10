// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/reorg_yolo.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "ie_core.hpp"
#include "single_layer_tests/reorg_yolo.hpp"

namespace LayerTestsDefinitions {

std::string reorgYoloLayerTest::getTestCaseName(testing::TestParamInfo<reorgYoloParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    size_t strides;
    std::string targetDevice;
    std::tie(inputShape, netPrecision, strides, targetDevice) = obj.param;

    std::ostringstream result;
    result << "inPRC=" << netPrecision.name() << "_";
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "strides=" << strides << "_";
    result << "targetDevice=" << targetDevice;

    return result.str();
}

void reorgYoloLayerTest::SetUp() {
    SetRefMode(LayerTestsUtils::RefMode::IE);
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    size_t stride;
    std::tie(inputShape, netPrecision, stride, targetDevice) = this->GetParam();

    ngraph::Strides strides({stride});
    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    const auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    const auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    const auto reorgYolo = std::make_shared<ngraph::opset4::ReorgYolo>(paramOuts[0], strides);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reorgYolo)};
    function = std::make_shared<ngraph::Function>(results, params, "reorgYolo");
}

TEST_P(reorgYoloLayerTest, CompareWithRefs) { Run(); }

}  // namespace LayerTestsDefinitions
