// Copyright (C) 2020 Intel Corporation
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
    typedef std::tuple<
            InferenceEngine::Precision,         // Network precision
            std::vector<std::vector<size_t>>,   // Input shapes
            std::string                        // Device name
            > matmulParams;

class MatMulLayerTest : public LayerTestsUtils::LayerTestsCommonClass<matmulParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<matmulParams> &obj);
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
