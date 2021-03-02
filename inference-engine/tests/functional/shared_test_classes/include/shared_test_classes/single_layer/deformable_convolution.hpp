// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {
typedef std::tuple<
        InferenceEngine::SizeVector,  // Kernel size
        InferenceEngine::SizeVector,  // Strides
        std::vector<int64_t>,         // Pad begin
        std::vector<int64_t>,         // Pad end
        InferenceEngine::SizeVector,  // Dilation
        size_t,                       // Num out channel
        size_t,                       // Group
        size_t,                       // Deformable group
        ngraph::op::PadType           // Padding type
> deformableConvSpecificParams;
typedef std::tuple<
        deformableConvSpecificParams,
        InferenceEngine::Precision,    // Net precision
        InferenceEngine::SizeVector,   // Input shapes
        InferenceEngine::SizeVector,   // Deformable shape
        LayerTestsUtils::TargetDevice  // Device name
> deformableConvLayerTestParamsSet;

class DeformableConvolutionLayerTest : public testing::WithParamInterface<deformableConvLayerTestParamsSet>,
                                       virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<deformableConvLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
