// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/space_to_batch.hpp"

using LayerTestsDefinitions::SpaceToBatchLayerTest;
using LayerTestsDefinitions::spaceToBatchParamsTuple;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::I32,
};

const std::vector<std::vector<size_t>> blockShapes = {
    {1, 2, 2, 1}
};

const std::vector<std::vector<size_t>> pads_begins = {
    {0, 0, 0, 0}
};

const std::vector<std::vector<size_t>> pads_ends = {
    {0, 0, 0, 0}
};

const std::vector<std::vector<size_t>> inputShapes = {
    {1, 2, 2, 1},  //
    {1, 4, 4, 1},  //
};

INSTANTIATE_TEST_CASE_P(SpaceToBatchSmokeCheck, SpaceToBatchLayerTest,
                        ::testing::Combine(::testing::ValuesIn(blockShapes),
                                           ::testing::ValuesIn(pads_begins),
                                           ::testing::ValuesIn(pads_ends),
                                           ::testing::ValuesIn(inputShapes),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        SpaceToBatchLayerTest::getTestCaseName);

}  // namespace
