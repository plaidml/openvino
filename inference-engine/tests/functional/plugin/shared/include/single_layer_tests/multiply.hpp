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
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "common_test_utils/test_constants.hpp"

namespace LayerTestsDefinitions {

using MultiplyParamsTuple = typename std::tuple<
        std::vector<std::vector<size_t>>, //input shapes
        InferenceEngine::Precision,       //Network precision
        std::string>;                     //Device name

class MultiplyLayerTest : public LayerTestsUtils::LayerTestsCommonClass<MultiplyParamsTuple>{
public:
    std::shared_ptr<ngraph::Function> fn;
    static std::string getTestCaseName(const testing::TestParamInfo<MultiplyParamsTuple> &obj);
protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
