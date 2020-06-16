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

// n.b. not testing pad_value because it is not compatible with all pad_modes and has a default
typedef std::tuple<
    std::vector<size_t>,            // pads_begin
    std::vector<size_t>,            // pads_end
    ngraph::op::PadMode,            // pad_mode
    InferenceEngine::Precision,     // Net precision
    std::vector<size_t>,            // Input shapes
    std::string                     // Target device name
> padParams;

class PadLayerTest : public LayerTestsUtils::LayerTestsCommonClass<padParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<padParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
