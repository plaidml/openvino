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

#include "single_layer_tests/clamp.hpp"

namespace LayerTestsDefinitions {

std::string ClampLayerTest::getTestCaseName(testing::TestParamInfo<clampParams> obj) {
    double min;
    double max;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(min, max, netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "MIN_" << min << "_";
    result << "MAX_" << max << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void ClampLayerTest::SetUp() {
    double min;
    double max;
    std::vector<size_t> inputShape;
    std::tie(min, max, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto clamp = std::make_shared<ngraph::opset1::Clamp>(paramOuts.at(0), min, max);

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(clamp));
    fnPtr = std::make_shared<ngraph::Function>(results, params, "clamp");
}

TEST_P(ClampLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
