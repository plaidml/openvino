// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <ie_core.hpp>

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "single_layer_tests/mat_mul.hpp"

namespace LayerTestsDefinitions {

std::string MatMulLayerTest::getTestCaseName(testing::TestParamInfo<matmulParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::vector<InferenceEngine::SizeVector> inputShapes;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void MatMulLayerTest::SetUp() {
    std::vector<InferenceEngine::SizeVector> inputShapes;
    std::tie(netPrecision, inputShapes, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, {inputShapes});
    auto paramIn = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    IE_ASSERT(paramIn.size() == 2);
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(paramsIn[0], paramsIn[1]);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(matmul)};
    fnPtr = std::make_shared<ngraph::Function>(results, paramsIn, "MatdsakjfldsakjfMul");
}

TEST_P(MatMulLayerTest, CompareWithRefs) {
    inferAndValidate();
}
}  // namespace LayerTestsDefinitions
