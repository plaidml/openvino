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

#include "single_layer_tests/reduce_max.hpp"

namespace LayerTestsDefinitions {

std::string ReduceMaxLayerTest::getTestCaseName(testing::TestParamInfo<reduceMaxParams> obj) {
    std::vector<int64_t> axes;
    bool keep_dims;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(axes, keep_dims, netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "IO_" << CommonTestUtils::vec2str(axes) << "_";
    result << "KD_" << keep_dims << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void ReduceMaxLayerTest::SetUp() {
    std::vector<int64_t> axes;
    bool keep_dims;
    std::vector<size_t> inputShape;
    std::tie(axes, keep_dims, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto AxesOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{axes.size()},
                                                                  axes);
    const auto reduce_max = std::make_shared<ngraph::opset1::ReduceMax>(paramOuts.at(0), AxesOp);

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(reduce_max));
    fnPtr = std::make_shared<ngraph::Function>(results, params, "reduce_max");
}

TEST_P(ReduceMaxLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
