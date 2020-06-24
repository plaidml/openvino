// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

#include "subgraph_tests/split_conv_concat.hpp"


namespace LayerTestsDefinitions {

std::string SplitConvConcat::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj) {
    InferenceEngine::Precision inputPrecision, netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    std::string targetDevice;
    std::tie(inputPrecision, netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "inPRC_" << inputPrecision.name() << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void SplitConvConcat::SetUp() {
    std::vector<size_t> inputShape;
    std::tie(inputPrecision, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto conv1 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(conv1);

    auto conv2 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv2);
    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), relu2->output(0)}, 1);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
    fnPtr = std::make_shared<ngraph::Function>(results, params, "SplitConvConcat");
}

TEST_P(SplitConvConcat, CompareWithRefImpl) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions