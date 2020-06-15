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
    int64_t,                        // batch_axis
    int64_t,                        // seq_axis
    std::vector<size_t>,            // lengths
    InferenceEngine::Precision,     // Net precision
    std::vector<size_t>,            // Input shapes
    std::string                     // Target device name
> reverseSequenceParams;

class ReverseSequenceLayerTest : public LayerTestsUtils::LayerTestsCommonClass<reverseSequenceParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<reverseSequenceParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
