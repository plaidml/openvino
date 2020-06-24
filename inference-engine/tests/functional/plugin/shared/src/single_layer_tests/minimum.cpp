// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <debug.h>
#include "ie_core.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "single_layer_tests/minimum.hpp"

namespace LayerTestsDefinitions {
    std::string MinimumLayerTest::getTestCaseName(const testing::TestParamInfo<MinimumParamsTuple> &obj) {
        std::vector<std::vector<size_t>> inputShapes;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::tie(inputShapes, netPrecision, targetName) = obj.param;
        std::ostringstream results;

        results << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
        results << "netPRC_" << netPrecision.name() << "_";
        results << "targetDevice_" << targetName << "_";
        return results.str();
    }

    void MinimumLayerTest::SetUp() {
        std::vector<std::vector<size_t>> inputShapes;
        std::tie(inputShapes, netPrecision, targetDevice) = this->GetParam();
        const std::size_t inputDim = InferenceEngine::details::product(inputShapes[0]);
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        std::vector<size_t> shapeInput{1, inputDim};
        auto input = ngraph::builder::makeParams(ngPrc, {shapeInput});
        auto constMul = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{1}, std::vector<float>{-1.0f});
        auto minimum = std::make_shared<ngraph::opset1::Minimum>(input[0], constMul);
        fnPtr = std::make_shared<ngraph::Function>(minimum, input, "minimum");
    }

    TEST_P(MinimumLayerTest, CompareWithRefs){
        inferAndValidate();
    };
} // namespace LayerTestsDefinitions
