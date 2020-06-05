// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "single_layer_tests/extract_image_patches.hpp"

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"


namespace LayerTestsDefinitions {

std::string ExtractImagePatchesTest::getTestCaseName(const testing::TestParamInfo<extractImagePatchesTuple> &obj) {
    std::vector<size_t> inputShape, kernel, strides, rates;
    ngraph::op::PadType pad_type;
    InferenceEngine::Precision netPrc;
    std::string targetName;
    std::tie(inputShape, kernel, strides, rates, pad_type, netPrc, targetName) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "netPRC_" << netPrc.name() << "_";
    result << "K_" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S_" << CommonTestUtils::vec2str(strides) << "_";
    result << "R_" << CommonTestUtils::vec2str(rates) << "_";
    result << "P_" << pad_type << "_";
    result << "targetDevice_" << targetName;
    return result.str();
}

void ExtractImagePatchesTest::SetUp() {
    std::vector<size_t> inputShape, kernel, strides, rates;
    ngraph::op::PadType pad_type;
    std::tie(inputShape, kernel, strides, rates, pad_type, netPrecision, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto inputNode = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(inputShape));
    ngraph::ParameterVector params = {inputNode};

    auto extImgPatches = std::make_shared<ngraph::opset3::ExtractImagePatches>(
        inputNode, ngraph::Shape(kernel), ngraph::Strides(strides), ngraph::Shape(rates), pad_type);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(extImgPatches)};
    fnPtr = std::make_shared<ngraph::Function>(results, params, "ExtractImagePatches");
}

TEST_P(ExtractImagePatchesTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
