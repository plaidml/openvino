// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/shuffle_channels.hpp"
using LayerTestsDefinitions::ShuffleChannelsLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inputShapes = {
    {5, 12, 200, 400},
    {1, 30, 40, 40}
};

const std::vector<shuffleChannelsSpecificParams> shuffleChannelsParams = {
    {1, 2},
    {1, 6}
};

INSTANTIATE_TEST_CASE_P(ShuffleChannelsSmokeCheck, ShuffleChannelsLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shuffleChannelsParams),                //
                                            ::testing::ValuesIn(netPrecisions),                        //
                                            ::testing::ValuesIn(inputShapes),                          //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),       //
                        ShuffleChannelsLayerTest::getTestCaseName);

}  // namespace
