// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

using ROIAlignParams = typename std::tuple<std::vector<size_t>,        // input shape N C H W
                                           size_t,                     // num_ROIs
                                           size_t,                     // pooled height
                                           size_t,                     // pooled width
                                           size_t,                     // sampling ratio
                                           float,                      // spatial scale
                                           std::string,                // mode of pooling
                                           InferenceEngine::Precision, // Network precision
                                           std::string>;               // Device name

class ROIAlignLayerTest : public testing::WithParamInterface<ROIAlignParams>,
                          virtual public LayerTestsUtils::LayerTestsCommon {
  public:
    static std::string getTestCaseName(const testing::TestParamInfo<ROIAlignParams> &obj);

  protected:
    void SetUp() override;
};

} // namespace LayerTestsDefinitions