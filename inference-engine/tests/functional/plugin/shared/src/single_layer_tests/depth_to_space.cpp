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

#include "single_layer_tests/depth_to_space.hpp"

// using namespace ngraph::opset1;
using ngraph::opset1::DepthToSpace;

namespace LayerTestsDefinitions {

static inline std::string DepthToSpaceModeToString(const DepthToSpace::DepthToSpaceMode& mode) {
    static std::map<DepthToSpace::DepthToSpaceMode, std::string> names = {
        {DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, "BLOCKS_FIRST"},
        {DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, "DEPTH_FIRST"},
    };

    auto i = names.find(mode);
    if (i != names.end())
        return i->second;
    else
        throw std::runtime_error("Unsupported DepthToSpaceMode");
}

std::string DepthToSpaceLayerTest::getTestCaseName(const testing::TestParamInfo<depthToSpaceParamsTuple> &obj) {
    std::vector<size_t> inShape;
    DepthToSpace::DepthToSpaceMode mode;
    std::size_t blockSize;
    InferenceEngine::Precision inputPrecision;
    std::string targetName;
    std::tie(inShape, inputPrecision, mode, blockSize, targetName) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inShape) << "_";
    result << "inPrc_" << inputPrecision.name() << "_";
    result << "M_" << DepthToSpaceModeToString(mode) << "_";
    result << "BS_" << blockSize << "_";
    result << "targetDevice_" << targetName << "_";
    return result.str();
}

void DepthToSpaceLayerTest::SetUp() {
    std::vector<size_t> inShape;
    DepthToSpace::DepthToSpaceMode mode;
    std::size_t blockSize;
    InferenceEngine::Precision inputPrecision;
    std::tie(inShape, inputPrecision, mode, blockSize, targetDevice) = this->GetParam();
    auto inPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto params = ngraph::builder::makeParams(inPrc, {inShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto s2d = ngraph::builder::makeDepthToSpace(paramOuts[0], mode, blockSize);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(s2d)};
    fnPtr = std::make_shared<ngraph::Function>(results, params, "DepthToSpace");
}

TEST_P(DepthToSpaceLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions
