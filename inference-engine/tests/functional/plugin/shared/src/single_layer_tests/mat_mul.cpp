// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "single_layer_tests/mat_mul.hpp"

namespace LayerTestsDefinitions {

std::string MatMulLayerTest::getTestCaseName(const testing::TestParamInfo<matmulParams> &obj) {
    transposeParams transParams;
    InferenceEngine::Precision netPrecision;
    std::vector<InferenceEngine::SizeVector> inputShapes;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(transParams, netPrecision, inputShapes, targetDevice) = obj.param;
    bool transA, transB;
    std::tie(transA,transB) = transParams;
    
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "AT_" << transA << "_";
    result << "BT_" << transB << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void MatMulLayerTest::SetUp() {
    transposeParams transParams;
    std::vector<InferenceEngine::SizeVector> inputShapes;
    std::tie(transParams, netPrecision, inputShapes, targetDevice) = this->GetParam();
    bool transA, transB;
    std::tie(transA, transB) = transParams;    
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, {inputShapes});
    auto paramIn = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    IE_ASSERT(paramIn.size() == 2);
    //auto matmul = std::dynamic_pointer_cast<ngraph::opset1::MatMul>(
    //        ngraph::builder::makeMatMul(paramIn[0], paramIn[1], ngPrc, transA, transB));
    auto matmul = std::make_shared<ngraph::opset1::MatMul>(paramsIn[0], paramsIn[1], transA, transB);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(matmul)};
    fnPtr = std::make_shared<ngraph::Function>(results, paramsIn, "MatMul");
}

TEST_P(MatMulLayerTest, CompareWithRefs) {
    inferAndValidate();
}
}  // namespace LayerTestsDefinitions
