// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
    double,                                    // epsilon
    InferenceEngine::Precision,                // Net precision
    std::vector<InferenceEngine::SizeVector>,  // Input shapes
    std::string                                // Target device name
> batchNormInferenceParams;

class BatchNormInferenceLayerTest : public LayerTestsUtils::LayerTestsCommonClass<batchNormInferenceParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<batchNormInferenceParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
