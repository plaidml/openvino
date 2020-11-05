// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/space_to_batch.hpp"

using LayerTestsDefinitions::SpaceToBatchLayerTest;
using LayerTestsDefinitions::spaceToBatchParamsTuple;

namespace {

spaceToBatchParamsTuple bts_only_test_cases[] = {
    spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {4, 1, 1, 1}, InferenceEngine::Precision::FP32,
                            CommonTestUtils::DEVICE_PLAIDML),
    spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {4, 3, 1, 1}, InferenceEngine::Precision::FP32,
                            CommonTestUtils::DEVICE_PLAIDML),
    spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {4, 1, 2, 2}, InferenceEngine::Precision::FP32,
                            CommonTestUtils::DEVICE_PLAIDML),
    spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {8, 1, 1, 2}, InferenceEngine::Precision::FP32,
                            CommonTestUtils::DEVICE_PLAIDML),
    spaceToBatchParamsTuple({1, 1, 3, 2, 2}, {0, 0, 1, 0, 3}, {0, 0, 2, 0, 0}, {12, 1, 2, 1, 2},
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
};

INSTANTIATE_TEST_CASE_P(SpaceToBatchSmokeCheck, SpaceToBatchLayerTest, ::testing::ValuesIn(bts_only_test_cases),
                        SpaceToBatchLayerTest::getTestCaseName);

}  // namespace
