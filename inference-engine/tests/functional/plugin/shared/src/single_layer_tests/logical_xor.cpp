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

#include "single_layer_tests/logical_xor.hpp"

namespace LayerTestsDefinitions {

std::string LogicalXorLayerTest::getTestCaseName(testing::TestParamInfo<logicalXorParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::vector<std::vector<size_t>> inputShapes;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void LogicalXorLayerTest::SetUp() {
    std::vector<InferenceEngine::SizeVector> inputShapes;
    std::tie(netPrecision, inputShapes, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto paramsIn = ngraph::builder::makeParams(ngPrc, {inputShapes});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    IE_ASSERT(paramOuts.size() == 2);
    const auto logical_xor = std::make_shared<ngraph::opset1::LogicalXor>(paramOuts.at(0), paramOuts.at(1));

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(logical_xor));
    fnPtr = std::make_shared<ngraph::Function>(results, paramsIn, "logical_xor");
}

TEST_P(LogicalXorLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
