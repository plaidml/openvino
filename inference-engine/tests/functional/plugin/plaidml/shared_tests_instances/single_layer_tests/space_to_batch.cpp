// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/space_to_batch.hpp"

using LayerTestsDefinitions::SpaceToBatchLayerTest;
using LayerTestsDefinitions::spaceToBatchParamsTuple;

namespace {

spaceToBatchParamsTuple stb_only_test_cases[] = {
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 1, 2, 2},
                                InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 3, 2, 2},
                                InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 1, 4, 4},
                                InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
        spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 2}, {0, 0, 0, 0}, {2, 1, 2, 4},
                                InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
        spaceToBatchParamsTuple({1, 1, 3, 2, 2}, {0, 0, 1, 0, 3}, {0, 0, 2, 0, 0}, {1, 1, 3, 2, 1},
                                InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
};

INSTANTIATE_TEST_CASE_P(
        smoke_PlaidML, SpaceToBatchLayerTest, ::testing::ValuesIn(stb_only_test_cases),
        SpaceToBatchLayerTest::getTestCaseName);

}  // namespace
