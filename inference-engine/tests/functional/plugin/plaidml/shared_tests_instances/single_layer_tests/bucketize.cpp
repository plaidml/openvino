// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/bucketize.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

using LayerTestsDefinitions::BucketizeLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I8,
};

std::vector<std::vector<size_t>> input_shapes{{2, 2}};
std::vector<std::vector<size_t>> buckets_shapes{{3}};

// Output_type shall support i64 && i32, close i64 for openvino limitation
std::vector<std::string> output_type{/*"i64",*/ "i32"};
std::vector<bool> with_right_bound{true, false};

INSTANTIATE_TEST_CASE_P(BucketizeCheckIndex, BucketizeLayerTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(buckets_shapes),
                                            ::testing::ValuesIn(output_type),
                                            ::testing::ValuesIn(with_right_bound),
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         BucketizeLayerTest::getTestCaseName);
}  // namespace
