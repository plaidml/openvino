// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsDefinitions {

using softMaxLayerTestParams = std::tuple<
        InferenceEngine::Precision,         // netPrecision
        InferenceEngine::Layout,            // inputLayout
        InferenceEngine::SizeVector,        // inputShape
        size_t,                             // axis
        std::string                         // targetDevice
>;

class SoftMaxLayerTest : public LayerTestsUtils::LayerTestsCommonClass<softMaxLayerTestParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<softMaxLayerTestParams> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
