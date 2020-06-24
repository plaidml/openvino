// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/not_equal.hpp"

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <ie_core.hpp>


namespace LayerTestsDefinitions {

std::string NotEqualLayerTest::getTestCaseName(const testing::TestParamInfo<NotEqualTestParam>& obj) {
    InferenceEngine::Precision netPrecision;
    std::vector<InferenceEngine::SizeVector> inputShapes;
    std::string targetDevice;

    std::tie(inputShapes, netPrecision, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPrc_" << netPrecision.name() << "_";
    result << "targetDevice_" << targetDevice;

    return result.str();
}

void NotEqualLayerTest::SetUp() {
    std::vector<InferenceEngine::SizeVector> inputShapes;
    std::tie(inputShapes, netPrecision, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsVector = ngraph::builder::makeParams(ngPrc, {inputShapes});
    IE_ASSERT(paramsVector.size() == 2);

    auto notEqualOp = std::make_shared<ngraph::opset1::NotEqual>(paramsVector[0], paramsVector[1]);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(notEqualOp)};

    fnPtr = std::make_shared<ngraph::Function>(results, paramsVector, "NotEqual");
}

TEST_P(NotEqualLayerTest, CompareWithRefs) {
    inferAndValidate();
}
}  // namespace LayerTestsDefinitions
