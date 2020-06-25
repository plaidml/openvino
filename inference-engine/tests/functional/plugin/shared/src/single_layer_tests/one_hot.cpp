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

#include "single_layer_tests/one_hot.hpp"

namespace LayerTestsDefinitions {

std::string OneHotLayerTest::getTestCaseName(testing::TestParamInfo<oneHotParams> obj) {
    int64_t axis;
    size_t depth;
    float on_value;
    float off_value;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(axis, depth, on_value, off_value, netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "D_" << depth << "_";
    result << "ON_" << on_value << "_";
    result << "OFF_" << off_value << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    auto string = result.str();
    std::replace(string.begin(), string.end(), '-', '_');
    std::replace(string.begin(), string.end(), '.', '_');
    return string;
}

void OneHotLayerTest::SetUp() {
    int64_t axis;
    size_t depth;
    float on_value;
    float off_value;
    std::vector<size_t> inputShape;
    std::tie(axis, depth, on_value, off_value, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    std::vector<size_t> depth_vec;
    std::vector<float> on_value_vec;
    std::vector<float> off_value_vec;
    depth_vec.push_back(depth);
    on_value_vec.push_back(on_value);
    off_value_vec.push_back(off_value);

    const auto DepthOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{},
                                                                    depth_vec);
    const auto OnValueOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{},
                                                                      on_value_vec);
    const auto OffValueOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{},
                                                                       off_value_vec);
    const auto one_hot = std::make_shared<ngraph::opset1::OneHot>(paramOuts.at(0), DepthOp, OnValueOp, OffValueOp, axis);

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(one_hot));
    fnPtr = std::make_shared<ngraph::Function>(results, params, "one_hot");
}

TEST_P(OneHotLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
