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

#include "single_layer_tests/normalize_l2.hpp"

namespace LayerTestsDefinitions {

std::string NormalizeL2LayerTest::getTestCaseName(testing::TestParamInfo<normalizeL2Params> obj) {
    float eps;
    ngraph::op::EpsMode eps_mode;
    std::vector<int64_t> axes;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(eps, eps_mode, axes, netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "A_" << CommonTestUtils::vec2str(axes) << "_";
    result << "E_" << eps << "_";
    result << "EM_" << eps_mode << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    auto string = result.str();
    std::replace(string.begin(), string.end(), '-', '_');
    std::replace(string.begin(), string.end(), '.', '_');
    return string;
}

void NormalizeL2LayerTest::SetUp() {
    float eps;
    ngraph::op::EpsMode eps_mode;
    std::vector<int64_t> axes;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(eps, eps_mode, axes, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto AxesOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{axes.size()},
                                                                  axes);
    const auto normalize_l2 = std::make_shared<ngraph::opset1::NormalizeL2>(paramOuts.at(0), AxesOp, eps, eps_mode);

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(normalize_l2));
    function = std::make_shared<ngraph::Function>(results, params, "normalize_l2");
}

TEST_P(NormalizeL2LayerTest, CompareWithRefs) {
    Run();
};

}  // namespace LayerTestsDefinitions
