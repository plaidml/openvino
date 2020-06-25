// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/batch_norm_inference.hpp"

namespace LayerTestsDefinitions {

std::string BatchNormInferenceLayerTest::getTestCaseName(testing::TestParamInfo<batchNormInferenceParams> obj) {
    double epsilon;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(epsilon, netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "Eps_" << epsilon << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    auto string = result.str();
    std::replace(string.begin(), string.end(), '-', '_');
    std::replace(string.begin(), string.end(), '.', '_');
    return string;
}

void BatchNormInferenceLayerTest::SetUp() {
    double epsilon;
    std::vector<size_t> inputShape;
    std::tie(epsilon, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    IE_ASSERT(paramOuts.size() == 5);

    const auto batch_norm_inference = std::make_shared<ngraph::opset1::BatchNormInference>(paramOuts.at(0), paramOuts.at(1), paramOuts.at(2), paramOuts.at(3), paramOuts.at(4), epsilon);

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(batch_norm_inference));
    fnPtr = std::make_shared<ngraph::Function>(results, params, "batch_norm_inference");
}

TEST_P(BatchNormInferenceLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
