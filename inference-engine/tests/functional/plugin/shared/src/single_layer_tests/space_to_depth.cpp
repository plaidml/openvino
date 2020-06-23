// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <ie_core.hpp>
#include <ngraph_functions/builders.hpp>

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "common_test_utils/common_utils.hpp"

#include "single_layer_tests/space_to_depth.hpp"

// using namespace ngraph::opset1;
using ngraph::opset1::SpaceToDepth;

namespace LayerTestsDefinitions {

static inline std::string SpaceToDepthModeToString(const SpaceToDepth::SpaceToDepthMode& mode) {
    static std::map<SpaceToDepth::SpaceToDepthMode, std::string> names = {
        {SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST, "BLOCKS_FIRST"},
        {SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST, "DEPTH_FIRST"},
    };

    auto i = names.find(mode);
    if (i != names.end())
        return i->second;
    else
        throw std::runtime_error("Unsupported SpaceToDepthMode");
}

std::string SpaceToDepthLayerTest::getTestCaseName(const testing::TestParamInfo<spaceToDepthParamsTuple> &obj) {
    std::vector<size_t> inShape;
    SpaceToDepth::SpaceToDepthMode mode;
    std::size_t blockSize;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::tie(inShape, netPrecision, mode, blockSize, targetName) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inShape) << "_";
    result << "inPrc_" << netPrecision.name() << "_";
    result << "M_" << SpaceToDepthModeToString(mode) << "_";
    result << "BS_" << blockSize << "_";
    result << "targetDevice_" << targetName << "_";
    return result.str();
}

void SpaceToDepthLayerTest::SetUp() {
    std::vector<size_t> inShape;
    SpaceToDepth::SpaceToDepthMode mode;
    std::size_t blockSize;
    std::tie(inShape, netPrecision, mode, blockSize, targetDevice) = this->GetParam();
    auto inPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(inPrc, {inShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto s2d = ngraph::builder::makeSpaceToDepth(paramOuts[0], mode, blockSize);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(s2d)};
    fnPtr = std::make_shared<ngraph::Function>(results, params, "SpaceToDepth");
}

TEST_P(SpaceToDepthLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
