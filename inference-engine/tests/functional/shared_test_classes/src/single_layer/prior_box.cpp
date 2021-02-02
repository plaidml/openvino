// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/prior_box.hpp"

namespace LayerTestsDefinitions {
std::string PriorBoxLayerTest::getTestCaseName(const testing::TestParamInfo<priorBoxParams> &obj) {
    priorBoxAttrs specAttrs;
    std::vector<float> variance;
    bool scaleAllSizes;
    bool useFixedSizes;
    bool useFixedRatios;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape, imageShape;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(specAttrs, variance, scaleAllSizes, useFixedSizes, useFixedRatios, netPrecision, inputShape, imageShape, targetDevice,
             config) = obj.param;
    std::vector<float> minSize, maxSize, aspectRatio, density, fixedRatio, fixedSize;
    bool clip, flip;
    float step, offset;
    std::tie(minSize, maxSize, aspectRatio, density, fixedRatio, fixedSize, clip, flip, step, offset) = specAttrs;
    std::ostringstream result;
    result << "min_size=" << CommonTestUtils::vec2str(minSize) << "_";
    result << "max_size=" << CommonTestUtils::vec2str(maxSize) << "_";
    result << "aspect_ratio=" << CommonTestUtils::vec2str(aspectRatio) << "_";
    result << "density=" << CommonTestUtils::vec2str(density) << "_";
    result << "fixed_ratio=" << CommonTestUtils::vec2str(fixedRatio) << "_";
    result << "fixed_Size=" << CommonTestUtils::vec2str(fixedSize) << "_";
    result << "clip=" << clip << "_";
    result << "flip=" << flip << "_";
    result << "step=" << step << "_";
    result << "offset=" << offset << "_";
    result << "variance=" << CommonTestUtils::vec2str(variance) << "_";
    result << "scale_all_sizes=" << scaleAllSizes << "_";
    result << "use_fixed_size=" << useFixedSizes << "_";
    result << "use_fixed_ratio=" << useFixedRatios << "_";
    result << "net_precision=" << netPrecision.name() << "_";
    result << "input_shape=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "imageShape=" << CommonTestUtils::vec2str(imageShape) << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void PriorBoxLayerTest::SetUp() {
    priorBoxAttrs specAttrs;
    std::vector<float> variance;
    bool scaleAllSizes;
    bool useFixedSizes;
    bool useFixedRatios;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape, imageShape;
    std::tie(specAttrs, variance, scaleAllSizes, useFixedSizes, useFixedRatios, netPrecision, inputShape, imageShape, targetDevice,
             configuration) = this->GetParam();
    std::vector<float> minSize, maxSize, aspectRatio, density, fixedRatio, fixedSize;
    bool clip, flip;
    float step, offset;
    std::tie(minSize, maxSize, aspectRatio, density, fixedRatio, fixedSize, clip, flip, step, offset) = specAttrs;

    auto inputPrc = ngraph::element::Type_t::i32;
    std::shared_ptr<ngraph::opset1::Constant> layerShapeConstNode = std::make_shared<ngraph::opset1::Constant>(
            inputPrc, ngraph::Shape{inputShape.size()}, inputShape);

    auto paramsIn = ngraph::builder::makeParams(inputPrc, {inputShape, imageShape});
    auto paramIn = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));

    // priorBoxAttrs
    ngraph::op::PriorBoxAttrs attributes;
    if (useFixedSizes && fixedSize.size() > 0) {
        attributes.fixed_size = fixedSize;
        if (useFixedRatios && fixedRatio.size() > 0) {
            attributes.fixed_ratio = fixedRatio;
        } else {
            attributes.aspect_ratio = aspectRatio;
        }
        // the count of fixed_size and density shall be same for limit of ngraph
        // implementation, cut here
        int densityCount = density.size();
        int fixedCount = fixedSize.size();
        int dstCount = densityCount < fixedCount ? densityCount : fixedCount;
        IE_ASSERT(dstCount > 0);
        std::vector<float> dstFixedSize(fixedSize.begin(), fixedSize.begin() + dstCount);
        std::vector<float> dstDensity(density.begin(), density.begin() + dstCount);
        attributes.fixed_size = dstFixedSize;
        attributes.density = dstDensity;
    } else {
        attributes.min_size = minSize;
        if (scaleAllSizes) {
            attributes.max_size = maxSize;
        }
        attributes.aspect_ratio = aspectRatio;
    }
    attributes.clip = clip;
    attributes.flip = flip;
    attributes.step = step;
    attributes.offset = offset;
    attributes.variance = variance;
    attributes.scale_all_sizes = scaleAllSizes;
    auto priorBox = std::dynamic_pointer_cast<ngraph::opset4::PriorBox>(
            std::make_shared<ngraph::opset4::PriorBox>(layerShapeConstNode, paramIn[1], attributes));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(priorBox)};
    function = std::make_shared<ngraph::Function>(results, paramsIn, "PriorBox");
}

}  // namespace LayerTestsDefinitions
