// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/mat_mul.hpp"

using LayerTestsDefinitions::MatMulTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const auto params_ATranspose = ::testing::Combine(::testing::Values(true),  //
                                                  ::testing::Values(false)  //
);

const auto params_BTranspose = ::testing::Combine(::testing::Values(false),  //
                                                  ::testing::Values(true)    //
);

const std::vector<std::vector<std::size_t>> inputShapesA = {
    {1},        //
    {3, 3},  //
    {5, 5},  //
};

const std::vector<std::vector<std::size_t>> inputShapesB = {
    {1},        //
    {3, 1},  //
    {5, 2},  //
};

INSTANTIATE_TEST_CASE_P(MatMul_NoTranspose, MatMulTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::ValuesIn(inputShapesA),                     //
                                            ::testing::ValuesIn(inputShapesB),                     //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         MatMulTest::getTestCaseName);

}  // namespace
