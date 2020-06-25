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

#include "single_layer_tests/broadcast.hpp"

namespace LayerTestsDefinitions {


std::string BroadcastLayerTest::getTestCaseName(testing::TestParamInfo<broadcastParams> obj) {
    ngraph::op::AutoBroadcastType mode;
    std::vector<int64_t> target_shape;
    std::vector<int64_t> axes_mapping;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    std::tie(mode, target_shape, axes_mapping, netPrecision, inputShape, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "M_" << mode << "_";
    result << "TS_" << CommonTestUtils::vec2str(target_shape) << "_";
    result << "AM_" << CommonTestUtils::vec2str(axes_mapping) << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void BroadcastLayerTest::SetUp() {
    ngraph::op::AutoBroadcastType mode;
    std::vector<int64_t> target_shape;
    std::vector<int64_t> axes_mapping;
    InferenceEngine::SizeVector inputShape;
    std::tie(mode, target_shape, axes_mapping, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto TargetShapeOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                          ngraph::Shape{target_shape.size()},
                                                                          target_shape);
    const auto AxesMappingOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                          ngraph::Shape{axes_mapping.size()},
                                                                          axes_mapping);
    const auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(paramOuts.at(0), TargetShapeOp, AxesMappingOp, mode);

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(broadcast));
    fnPtr = std::make_shared<ngraph::Function>(results, params, "broadcast");
}

TEST_P(BroadcastLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
