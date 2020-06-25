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

#include "single_layer_tests/convert.hpp"

namespace LayerTestsDefinitions {

std::string ConvertLayerTest::getTestCaseName(testing::TestParamInfo<convertParams> obj) {
    InferenceEngine::Precision targetPrecision;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(targetPrecision, netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "targetPRC_" << targetPrecision.name() << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void ConvertLayerTest::SetUp() {
    InferenceEngine::Precision targetPrecision;
    std::vector<size_t> inputShape;
    std::tie(targetPrecision, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto ngTargetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(targetPrecision);
    const auto convert = std::make_shared<ngraph::opset1::Convert>(paramOuts.at(0), ngTargetPrc);

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(convert));
    fnPtr = std::make_shared<ngraph::Function>(results, params, "convert");
}

TEST_P(ConvertLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
