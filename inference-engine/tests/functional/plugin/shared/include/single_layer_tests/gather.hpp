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

// Not using tensors for on/off_value b/c I'm worried they would take the same
// value, which would not be an effective test. So using floats.
// I suspect that this means precision must be F32
typedef std::tuple<
    std::vector<int64_t>,           // axis
    InferenceEngine::Precision,     // Net precision
    std::vector<size_t>,            // Input shapes
    std::string                     // Target device name
> gatherParams;

class GatherLayerTest : public LayerTestsUtils::LayerTestsCommonClass<gatherParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<gatherParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
