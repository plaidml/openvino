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

std::string ROIAlignLayerTest::getTestCaseName(const testing::TestParamInfo<ROIAlignParams> &obj) {
    std::vector<size_t> inputShape;
    size_t numROIs;
    size_t pooledH;
    size_t pooledW;
    size_t samplingRatio;
    float spatialScale;
    std::string mode;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(inputShape, numROIs, pooledH, pooledW, samplingRatio, spatialScale, mode, netPrecision, targetDevice) = obj.param;
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
    std::vector<size_t> inputShape;
    size_t numROIs;
    size_t pooledH;
    size_t pooledW;
    size_t samplingRatio;
    float spatialScale;
    std::string mode;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;

    std::tie(inputShape, numROIs, pooledH, pooledW, samplingRatio, spatialScale, mode, netPrecision, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape, {numROIs, 4}, {numROIs}});
    auto inputNodes = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes(params));

    auto roiAlign =
        std::make_shared<ngraph::opset3::ROIAlign>(
            inputNodes[0], inputNodes[1], inputNodes[2], pooledH, pooledW, samplingRatio, spatialScale, mode);
}

TEST_P(ROIAlignLayerTest, CompareWithRefs) {
    Run();
}

} // namespace LayerTestsDefinitions