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
        InferenceEngine::SizeVector,
        InferenceEngine::SizeVector,
        std::vector<ptrdiff_t>,
        std::vector<ptrdiff_t>,
        InferenceEngine::SizeVector,
        size_t,
        size_t,
        ngraph::op::PadType> groupConvSpecificParams;
typedef std::tuple<
        groupConvSpecificParams,
        InferenceEngine::Precision,
        InferenceEngine::SizeVector,
        std::string> groupConvLayerTestParamsSet;

namespace LayerTestsDefinitions {

class GroupConvolutionLayerTest : public LayerTestsUtils::LayerTestsCommonClass<groupConvLayerTestParamsSet> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<groupConvLayerTestParamsSet> obj);

protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions