// Copyright (C) 2020 Intel Corporation
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

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using roiPoolingParamsTuple = std::tuple<
        InferenceEngine::SizeVector,                // Input shape
        InferenceEngine::SizeVector,                // Coords shape
        std::vector<size_t>,                        // Pooled shape {pooled_h, pooled_w}
        float,                                      // Spatial scale
        ngraph::helpers::ROIPoolingTypes,           // ROIPooling method
        InferenceEngine::Precision,                 // Net precision
        ngraph::helpers::InputLayerType,            // Secondary input type
        LayerTestsUtils::TargetDevice>;             // Device name

class ROIPoolingLayerTest : public testing::WithParamInterface<roiPoolingParamsTuple>,
                            virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<roiPoolingParamsTuple> obj);
    void Infer() override;

protected:
    void SetUp() override;
    void GenerateCoords(const std::vector<size_t> &feat_map_shape, float *buffer, size_t size);

private:
    ngraph::helpers::ROIPoolingTypes pool_method;
    ngraph::helpers::InputLayerType secondaryInputType;
    float spatial_scale;
};

}  // namespace LayerTestsDefinitions
