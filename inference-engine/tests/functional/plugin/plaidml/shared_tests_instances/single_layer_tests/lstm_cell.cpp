// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/lstm_cell.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using LayerTestsDefinitions::LSTMCellTest;

namespace {
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<bool> decomposes = {false};
const std::vector<size_t> batches = {1};
const std::vector<size_t> hidden_sizes = {128};
const std::vector<size_t> input_sizes = {16};
const std::vector<std::vector<std::string>> activations = {
    {"sigmoid", "tanh", "tanh"}};
const std::vector<float> clips = {0.f};

INSTANTIATE_TEST_CASE_P(
    LSTMCell_default, LSTMCellTest,
    ::testing::Combine(::testing::ValuesIn(decomposes),  //
                       ::testing::ValuesIn(batches),     //
                       ::testing::ValuesIn(hidden_sizes),
                       ::testing::ValuesIn(input_sizes),
                       ::testing::ValuesIn(activations),
                       ::testing::ValuesIn(clips),
                       ::testing::ValuesIn(netPrecisions),  //
                       ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
    LSTMCellTest::getTestCaseName);
}  // namespace
