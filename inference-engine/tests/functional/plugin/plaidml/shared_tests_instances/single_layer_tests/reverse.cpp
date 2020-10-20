// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reverse.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

using LayerTestsDefinitions::ReverseLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16,
    // InferenceEngine::Precision::I64
};

std::vector<std::vector<size_t>> shape{{4, 6, 5}, {3, 9, 2}, {1, 4, 2, 1, 1, 3}};
std::vector<std::vector<size_t>> axes{{0}, {1}, {0, 2}};
std::vector<std::string> mode{"index", "mask"};

INSTANTIATE_TEST_CASE_P(ReverseCheck, ReverseLayerTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),                          //
                                            ::testing::ValuesIn(shape),                                  //
                                            ::testing::ValuesIn(axes),                                   //
                                            ::testing::ValuesIn(mode),                                   //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                                            ::testing::Values(std::map<std::string, std::string>({}))),  //
                         ReverseLayerTest::getTestCaseName);
}  // namespace
