// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <functional_test_utils/layer_test_utils.hpp>

#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

typedef std::tuple<
        std::vector<std::vector<size_t>>,  // mask, then, else shapes
        InferenceEngine::Precision,        // then, else precision
        ngraph::op::AutoBroadcastSpec,     // broadcast
        std::string> selectTestParams;     // device name

class SelectLayerTest : public LayerTestsUtils::LayerTestsCommonClass<selectTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo <selectTestParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions