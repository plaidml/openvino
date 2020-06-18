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

#include "single_layer_tests/selu.hpp"

namespace LayerTestsDefinitions {

std::string SeluLayerTest::getTestCaseName(testing::TestParamInfo<seluParams> obj) {
    double alpha;
    double lambda;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(alpha, lambda, netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "A_" << alpha << "_";
    result << "L_" << lambda << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void SeluLayerTest::SetUp() {
    double alpha;
    double lambda;
    std::vector<size_t> inputShape;
    std::tie(alpha, lambda, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    std::vector<double> alpha_vec;
    std::vector<double> lambda_vec;
    alpha_vec.push_back(alpha);
    lambda_vec.push_back(lambda);

    const auto AlphaOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{1},
                                                                      alpha_vec);
    const auto LambdaOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{1},
                                                                       lambda_vec);
    const auto selu = std::make_shared<ngraph::opset1::Selu>(paramOuts.at(0), AlphaOp, LambdaOp);

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(selu));
    fnPtr = std::make_shared<ngraph::Function>(results, params, "selu");
}

TEST_P(SeluLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
