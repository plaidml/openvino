// Copyright (C) 2020 Intel Corporation
//
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

using reorgYoloParams = std::tuple<
    InferenceEngine::SizeVector,  // inputShape
    InferenceEngine::Precision,   // netPrecision
    size_t,                       // strides
    std::string                  // targetDevice
>;

class reorgYoloLayerTest : public testing::WithParamInterface<reorgYoloParams>,
                           virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<reorgYoloParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
