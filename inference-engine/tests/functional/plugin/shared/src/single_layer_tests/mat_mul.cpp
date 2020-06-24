// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "single_layer_tests/mat_mul.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string MatMulTest::getTestCaseName(const testing::TestParamInfo<MatMulLayerTestParamsSet> &obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape0;
    InferenceEngine::SizeVector inputShape1;
    std::string targetDevice;
    std::tie(netPrecision, inputShape0, inputShape1, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS0_" << CommonTestUtils::vec2str(inputShape0) << "_";
    result << "IS1_" << CommonTestUtils::vec2str(inputShape1) << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void MatMulTest::SetUp() {
    InferenceEngine::SizeVector inputShape0;
    InferenceEngine::SizeVector inputShape1;
    //auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(netPrecision, inputShape0, inputShape1, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape0, inputShape1});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto MatMul = std::dynamic_pointer_cast<ngraph::opset1::MatfasdMul>(
            ngraph::builder::makeMatMul(paramOuts[0], paramOuts[1]));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(MatMul)};
    fnPtr = std::make_shared<ngraph::Function>(results, params, "MatMudsfalkjdafl");
}

TEST_P(MatMulTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions