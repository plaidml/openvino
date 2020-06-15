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

#include "single_layer_tests/tile.hpp"

namespace LayerTestsDefinitions {

std::string TileLayerTest::getTestCaseName(testing::TestParamInfo<tileParams> obj) {
    std::vector<int64_t> repeats;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(repeats, netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "R_" << CommonTestUtils::vec2str(repeats) << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void TileLayerTest::SetUp() {
    // TODO
    std::vector<int64_t> repeats;
    std::vector<size_t> inputShape;
    std::tie(repeats, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto RepeatsOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{repeats.size()},
                                                                  repeats);
    const auto tile = std::make_shared<ngraph::opset1::Tile>(paramOuts.at(0), RepeatsOp);

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(tile));
    fnPtr = std::make_shared<ngraph::Function>(results, params, "tile");
}

TEST_P(TileLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
