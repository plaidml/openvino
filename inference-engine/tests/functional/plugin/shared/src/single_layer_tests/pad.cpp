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

#include "single_layer_tests/pad.hpp"

namespace LayerTestsDefinitions {

std::string PadLayerTest::getTestCaseName(testing::TestParamInfo<padParams> obj) {
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;
    ngraph::op::PadMode pad_mode;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(pads_begin, pads_end, pad_mode, netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "PB_" << CommonTestUtils::vec2str(pads_begin) << "_";
    result << "PE_" << CommonTestUtils::vec2str(pads_end) << "_";
    result << "PM_" << pad_mode << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void PadLayerTest::SetUp() {
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;
    ngraph::op::PadMode pad_mode;
    InferenceEngine::SizeVector inputShape;
    std::tie(pads_begin, pads_end, pad_mode, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto PadsBeginOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{pads_begin.size()},
                                                                        pads_begin);
    const auto PadsEndOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{pads_end.size()},
                                                                      pads_end);
    const auto pad = std::make_shared<ngraph::opset1::Pad>(paramOuts.at(0), PadsBeginOp, PadsEndOp, pad_mode);

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(pad));
    fnPtr = std::make_shared<ngraph::Function>(results, params, "pad");
}

TEST_P(PadLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
