// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <ie_plugin_config.hpp>
#include <ie_core.hpp>
#include <functional>

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "single_layer_tests/bucketize.hpp"
#include <iostream>

namespace LayerTestsDefinitions {
    std::string BucketizeLayerTest::getTestCaseName(testing::TestParamInfo<bucketizeParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::vector<size_t> buckets;
    std::string output_type;
    bool with_right_bound;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(netPrecision, inputShapes, buckets, output_type, with_right_bound, targetDevice, config) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "Bucket=" << CommonTestUtils::vec2str(buckets) << "_";
    result << "output_type=" << output_type << "_";
    result << "with_right_bound=" << with_right_bound << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void BucketizeLayerTest::SetUp() {
    //SetRefMode(LayerTestsUtils::RefMode::CONSTANT_FOLDING);
    //SetRefMode(LayerTestsUtils::RefMode::INTERPRETER_TRANSFORMATIONS);
    //SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);
    SetRefMode(LayerTestsUtils::RefMode::IE);
    std::vector<size_t> inputShapes;
    std::vector<size_t> buckets;
    std::string output_type;
    bool with_right_bound;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, inputShapes, buckets, output_type, with_right_bound, targetDevice, configuration) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto outputType = (output_type == "i32") ? ngraph::element::Type_t::i32 : ngraph::element::Type_t::i64;
    auto paramsIn = ngraph::builder::makeParams(ngPrc, {inputShapes, buckets});
    auto paramIn = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramsIn));
    auto bucketize = std::dynamic_pointer_cast<ngraph::opset3::Bucketize>(
            std::make_shared<ngraph::opset3::Bucketize>(paramIn[0], paramIn[1], outputType, with_right_bound));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(bucketize)};
    function = std::make_shared<ngraph::Function>(results, paramsIn, "Bucketize");
    //std::cout << "function output element type:" << function->get_output_element_type(0) << std::endl;
    //auto shape = function->output(0).get_node()->get_output_shape(0);
    //std::cout << "function output shape:";
    //for (int i =0; i < shape.size(); i++) {
    //    std::cout << " " << shape[i];
    //}
    //std::cout << std::endl;
}

TEST_P(BucketizeLayerTest, CompareWithRefsDynamicBath) {
    Run();
}
}  // namespace LayerTestsDefinitions
