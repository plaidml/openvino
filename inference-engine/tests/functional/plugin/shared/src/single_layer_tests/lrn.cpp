// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "single_layer_tests/lrn.hpp"

namespace LayerTestsDefinitions {

std::string LrnLayerTest::getTestCaseName(testing::TestParamInfo<lrnLayerTestParamsSet> obj) {
    double alpha;
    size_t beta, bias, size;
    InferenceEngine::Precision  netPrecision;
    std::vector<size_t> inputShapes;
    std::string targetDevice;
    std::tie(alpha, beta, bias, size, netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << separator;
    result << "Alpha=" << alpha << separator;
    result << "Beta=" << beta << separator;
    result << "Bias=" << bias << separator;
    result << "Size=" << size << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "targetDevice=" << targetDevice;

    auto string = result.str();
    std::replace(string.begin(), string.end(), '-', '_');
    std::replace(string.begin(), string.end(), '.', '_');
    std::replace(string.begin(), string.end(), '(', '_');
    std::replace(string.begin(), string.end(), ')', '_');
    std::replace(string.begin(), string.end(), '=', '_');
    return string;
}

void LrnLayerTest::SetUp() {
    std::vector<size_t> inputShapes;
    auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
    size_t alpha, beta, bias, size;
    std::tie(alpha, beta, bias, size, netPrecision, inputShapes, targetDevice) = GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes});
    auto paramIn =
        ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto lrn = std::make_shared<ngraph::opset1::LRN>(paramIn[0], alpha, beta, bias, size);
    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(lrn)};
    function = std::make_shared<ngraph::Function>(results, params, "lrn");
}

TEST_P(LrnLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
