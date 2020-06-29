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

typedef std::tuple<
        bool,
        bool> transposeParams;
typedef std::tuple<
        transposeParams,
        InferenceEngine::Precision,
        InferenceEngine::SizeVector,
        std::string> matmulParams;

namespace LayerTestsDefinitions {

class MatMulLayerTest : public LayerTestsUtils::LayerTestsCommonClass<matmulParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<matmulParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
