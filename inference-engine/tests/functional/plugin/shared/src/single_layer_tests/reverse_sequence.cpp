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

#include "single_layer_tests/reverse_sequence.hpp"

namespace LayerTestsDefinitions {

std::string ReverseSequenceLayerTest::getTestCaseName(testing::TestParamInfo<reverseSequenceParams> obj) {
    int64_t batch_axis;
    int64_t seq_axis;
    std::vector<size_t> lengths;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(batch_axis, seq_axis, lengths, netPrecision, inputShapes, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "BA_" << batch_axis << "_";
    result << "SA_" << seq_axis << "_";
    result << "L_" << CommonTestUtils::vec2str(lengths) << "_";
    result << "netPRC_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;
    return result.str();
}

void ReverseSequenceLayerTest::SetUp() {
    int64_t batch_axis;
    int64_t seq_axis;
    std::vector<size_t> lengths;
    std::vector<size_t> inputShape;
    std::tie(batch_axis, seq_axis, lengths, netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    // TODO
    const auto LengthsOp = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{lengths.size()},
                                                                  lengths);
    const auto reverse_sequence = std::make_shared<ngraph::opset1::ReverseSequence>(paramOuts.at(0), LengthsOp, batch_axis, seq_axis);

    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::opset1::Result>(reverse_sequence));
    fnPtr = std::make_shared<ngraph::Function>(results, params, "reverse_sequence");
}

TEST_P(ReverseSequenceLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
