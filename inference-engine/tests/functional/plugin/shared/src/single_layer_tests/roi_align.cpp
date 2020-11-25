// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

#include "single_layer_tests/roi_align.hpp"

namespace LayerTestsDefinitions {

template <template <typename...> class R = std::vector, typename Top, typename Sub = typename Top::value_type>
R<typename Sub::value_type> flatten(Top const &all) {
    R<typename Sub::value_type> accum;
    for (auto &sub : all)
        accum.insert(std::end(accum), std::begin(sub), std::end(sub));

    return accum;
}

std::string ROIAlignLayerTest::getTestCaseName(const testing::TestParamInfo<ROIAlignParams> &obj) {
    ROIAlignSpecificParams roiParams;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(roiParams, netPrecision, targetDevice) = obj.param;
    InferenceEngine::SizeVector inputShape;
    std::vector<std::vector<size_t>> rois;
    std::vector<size_t> batchIndices;
    size_t numROIs;
    size_t pooledH;
    size_t pooledW;
    size_t samplingRatio;
    float spatialScale;
    std::string mode;
    std::tie(inputShape, rois, batchIndices, numROIs, pooledH, pooledW, samplingRatio, spatialScale, mode) = roiParams;

    std::ostringstream result;
    result << "inputShape=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "numROIs=" << numROIs << "_";
    result << "pooledH=" << pooledH << "_";
    result << "pooledW=" << pooledW << "_";
    result << "samplingRatio=" << samplingRatio << "_";
    result << "spatialScale=" << spatialScale << "_";
    result << "mode=" << mode << "_";
    result << "netPRC=" << netPrecision << "_";
    result << "targetDevice=" << targetDevice << "_";
    return result.str();
}

void ROIAlignLayerTest::SetUp() {
    // Use IE ref mode as ngraph can not run this operation now
    SetRefMode(LayerTestsUtils::RefMode::IE);

    ROIAlignSpecificParams roiParams;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(roiParams, netPrecision, targetDevice) = this->GetParam();
    InferenceEngine::SizeVector inputShape;
    std::vector<std::vector<size_t>> rois;
    std::vector<size_t> batchIndices;
    size_t numROIs;
    size_t pooledH;
    size_t pooledW;
    size_t samplingRatio;
    float spatialScale;
    std::string mode;
    std::tie(inputShape, rois, batchIndices, numROIs, pooledH, pooledW, samplingRatio, spatialScale, mode) = roiParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto flatRois = flatten(rois);
    ngraph::Shape roisShape = {flatRois.size()};
    auto roisNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, roisShape, flatRois.data());
    ngraph::Shape batchIndicesShape = {batchIndices.size()};
    auto batchIndicesNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, batchIndicesShape, batchIndices.data());
    auto roiAlign = std::make_shared<ngraph::opset3::ROIAlign>(paramOuts[0], roisNode, batchIndicesNode, pooledH, pooledW, samplingRatio, spatialScale, mode);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(roiAlign)};
    function = std::make_shared<ngraph::Function>(results, params, "ROIAlign");
}

TEST_P(ROIAlignLayerTest, CompareWithRefs) {
    Run();
}

} // namespace LayerTestsDefinitions