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

#include "ngraph/op/squared_difference.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

#include "single_layer_tests/squared_difference.hpp"

namespace LayerTestsDefinitions {

std::string SquaredDifferenceLayerTest::getTestCaseName(testing::TestParamInfo<squaredDifferenceParams> obj) {
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

void SquaredDifferenceLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::vector<InferenceEngine::SizeVector> inputShapes;
    std::tie(netPrecision, inputShapes, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes});

    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    IE_ASSERT(paramOuts.size() == 2);
    const auto squared_difference = std::make_shared<ngraph::opset1::SquaredDifference>(paramOuts.at(0), paramOuts.at(1));

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(squared_difference));
    function = std::make_shared<ngraph::Function>(results, params, "squared_difference");
}

TEST_P(SquaredDifferenceLayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
