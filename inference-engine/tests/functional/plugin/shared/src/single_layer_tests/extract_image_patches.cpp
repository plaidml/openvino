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
    ngraph::op::PadType padType;
    InferenceEngine::Precision netPrc;
    std::string targetName;
    std::tie(inputShape, kernel, strides, rates, padType, netPrc, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "netPRC=" << netPrc.name() << "_";
    result << "K=" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S=" << CommonTestUtils::vec2str(strides) << "_";
    result << "R=" << CommonTestUtils::vec2str(rates) << "_";
    result << "P=" << padType << "_";
    result << "targetDevice=" << targetName;
    return result.str();
}

void ExtractImagePatchesTest::SetUp() {
  std::vector<size_t> inputShape;
  InferenceEngine::SizeVector kernel, strides, rates;
  ngraph::op::PadType padType;
  InferenceEngine::Precision netPrecision;
  std::tie(inputShape, kernel, strides, rates, padType, netPrecision,
           targetDevice) = this->GetParam();
  auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
  auto paramsIn = ngraph::builder::makeParams(ngPrc, {inputShape});
  auto paramOut = ngraph::helpers::convert2OutputVector(
          ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
  auto extImgPatches = std::dynamic_pointer_cast<ngraph::opset3::ExtractImagePatches>(
          std::make_shared<ngraph::opset3::ExtractImagePatches>(paramOut[0], ngraph::Shape(kernel),
                                                            ngraph::Strides(strides), ngraph::Shape(rates), padType));
  ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(extImgPatches)};
  function = std::make_shared<ngraph::Function>(results, paramsIn, "ExtractImagePatches");
}

TEST_P(ExtractImagePatchesTest, CompareWithRefs) {
    Run();
};

} // namespace LayerTestsDefinitions
