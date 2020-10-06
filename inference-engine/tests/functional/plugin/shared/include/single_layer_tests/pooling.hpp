// Copyright (C) 2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

#include "functional_test_utils/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        ngraph::helpers::PoolingTypes,  // Pooling type, max or avg
        std::vector<size_t>,            // Kernel size
        std::vector<size_t>,            // Stride
        std::vector<size_t>,            // Pad begin
        std::vector<size_t>,            // Pad end
        ngraph::op::RoundingType,       // Rounding type
        ngraph::op::PadType,            // Pad type
        bool                            // Exclude pad
> poolSpecificParams;
typedef std::tuple<
        poolSpecificParams,
        InferenceEngine::Precision,     // Net precision
        std::vector<size_t>,            // Input shape
        std::string                     // Device name
> poolLayerTestParamsSet;

class PoolingLayerTest : public testing::WithParamInterface<poolLayerTestParamsSet>,
                         public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<poolLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions