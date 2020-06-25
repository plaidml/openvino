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

#include "single_layer_tests/mvn.hpp"

namespace LayerTestsDefinitions {

std::string MvnLayerTest::getTestCaseName(testing::TestParamInfo<mvnParams> obj) {
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::Precision netPrecision;
    bool acrossChannels, normalizeVariance;
    double eps;
    std::string targetDevice;
    std::tie(inputShapes, netPrecision, acrossChannels, normalizeVariance, eps, targetDevice) = obj.param;
    std::ostringstream result;
    result << "IS_" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "Precision_" << netPrecision.name() << "_";
    result << "AcrossChannels_" << (acrossChannels ? "TRUE" : "FALSE") << "_";
    result << "NormalizeVariance_" << (normalizeVariance ? "TRUE" : "FALSE") << "_";
    result << "EpsilonTimes1000000000_" << eps * 1000000000 << "_";
    result << "TargetDevice_" << targetDevice;
    auto string = result.str();
    std::replace(string.begin(), string.end(), '-', '_');
    std::replace(string.begin(), string.end(), '.', '_');
    return string;
}

void MvnLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShapes;
    bool acrossChanels, normalizeVariance;
    double eps;
    std::tie(inputShapes, netPrecision, acrossChanels, normalizeVariance, eps, targetDevice) = this->GetParam();
    auto inType = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto param = ngraph::builder::makeParams(inType, {inputShapes});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(param));
    auto mvn = std::dynamic_pointer_cast<ngraph::op::MVN>(ngraph::builder::makeMVN(paramOuts[0], acrossChanels, normalizeVariance, eps));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(mvn)};
    fnPtr = std::make_shared<ngraph::Function>(results, param, "mvn");
}

TEST_P(MvnLayerTest, CompareWithRefs) {
    inferAndValidate();
};

}  // namespace LayerTestsDefinitions