// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/space_to_batch.hpp"

using LayerTestsDefinitions::SpaceToBatchLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32  //
};

INSTANTIATE_TEST_CASE_P(smoke, SpaceToBatchLayerTest,
                         ::testing::Combine(::testing::Values(1, 1, 2, 2),                        //
                                            ::testing::Values(0, 0, 0, 0),                        //
                                            ::testing::Values(0, 0, 0, 0),                        //
                                            ::testing::Values(1, 1, 2, 2),                        //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         SpaceToBatchLayerTest::getTestCaseName);

}  // namespace
