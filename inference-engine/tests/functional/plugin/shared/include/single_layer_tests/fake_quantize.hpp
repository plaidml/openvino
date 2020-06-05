// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>
#include <memory>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

typedef std::tuple<
        size_t,             // levels
        std::vector<size_t> // const inputs shape
> fqSpecificParams;
typedef std::tuple<
        fqSpecificParams,
        InferenceEngine::Precision,   // Net precision
        InferenceEngine::SizeVector,  // Input shapes
        std::string // Device name
> fqLayerTestParamsSet;
namespace LayerTestsDefinitions {


class FakeQuantizeLayerTest : public LayerTestsUtils::LayerTestsCommonClass<fqLayerTestParamsSet> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<fqLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions
