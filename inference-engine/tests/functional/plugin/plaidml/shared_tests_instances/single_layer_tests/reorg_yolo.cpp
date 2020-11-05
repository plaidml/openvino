// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reorg_yolo.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

using LayerTestsDefinitions::reorgYoloLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
};

INSTANTIATE_TEST_CASE_P(reorgYoloCheck, reorgYoloLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>({10, 10, 10, 10})),   //
                                           ::testing::ValuesIn(netPrecisions),                         //
                                           ::testing::ValuesIn(std::vector<size_t>({1, 2, 5})),        //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),        //
                        reorgYoloLayerTest::getTestCaseName);
}  // namespace
