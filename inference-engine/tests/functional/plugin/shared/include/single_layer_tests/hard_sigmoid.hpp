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

// n.b. using floats for alpha & beta to get direct control; may break non-f32 precision
typedef std::tuple<
    float,                          // alpha
    float,                          // beta
    InferenceEngine::Precision,     // Net precision
    std::vector<size_t>,            // Input shapes
    std::string                     // Target device name
> hardSigmoidParams;

class HardSigmoidLayerTest : public LayerTestsUtils::LayerTestsCommonClass<hardSigmoidParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<hardSigmoidParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
