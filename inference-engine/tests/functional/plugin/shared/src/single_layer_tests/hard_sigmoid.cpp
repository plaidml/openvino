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

#include "single_layer_tests/hard_sigmoid.hpp"

namespace LayerTestsDefinitions {

std::string HardSigmoidLayerTest::getTestCaseName(testing::TestParamInfo<hardSigmoidParams> obj) {
    float alpha;
    float beta;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(alpha, beta, netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "A_" << alpha << "_";
    result << "B_" << beta << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void HardSigmoidLayerTest::SetUp() {
    float alpha;
    float beta;
    std::vector<size_t> inputShape;
    std::tie(alpha, beta, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    std::vector<float> alpha_vec;
    std::vector<float> beta_vec;
    alpha_vec.push_back(alpha);
    beta_vec.push_back(beta);

    const auto AlphaOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::f32, ngraph::Shape{},
                                                                    alpha_vec);
    const auto BetaOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::f32, ngraph::Shape{},
                                                                      beta_vec);
    const auto hard_sigmoid = std::make_shared<ngraph::opset1::HardSigmoid>(paramOuts.at(0), AlphaOp, BetaOp);

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(hard_sigmoid));
    fnPtr = std::make_shared<ngraph::Function>(results, params, "hard_sigmoid");
}

TEST_P(HardSigmoidLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
