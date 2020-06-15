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

#include "single_layer_tests/squeeze.hpp"

namespace LayerTestsDefinitions {

std::string SqueezeLayerTest::getTestCaseName(testing::TestParamInfo<squeezeParams> obj) {
    std::vector<int64_t> indexes_to_squeeze;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(indexes_to_squeeze, netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "ItS_" << CommonTestUtils::vec2str(indexes_to_squeeze) << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void SqueezeLayerTest::SetUp() {
    std::vector<int64_t> indexes_to_squeeze;
    std::vector<size_t> inputShape;
    std::tie(indexes_to_squeeze, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto IdxsToSqueezeOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{indexes_to_squeeze.size()},
                                                                  indexes_to_squeeze);
    const auto squeeze = std::make_shared<ngraph::opset1::Squeeze>(paramOuts.at(0), IdxsToSqueezeOp);

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(squeeze));
    fnPtr = std::make_shared<ngraph::Function>(results, params, "squeeze");
}

TEST_P(SqueezeLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
