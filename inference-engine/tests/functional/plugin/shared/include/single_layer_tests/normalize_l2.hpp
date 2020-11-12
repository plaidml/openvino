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
    float,                          // eps
    ngraph::op::EpsMode,            // eps_mode
    std::vector<int64_t>,           // axes
    InferenceEngine::Precision,     // Net precision
    std::vector<size_t>,            // Input shapes
    std::string                     // Target device name
> normalizeL2Params;

class NormalizeL2LayerTest : public testing::WithParamInterface<normalizeL2Params>,
                             virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<normalizeL2Params> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
