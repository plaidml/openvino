// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/roi_align.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using LayerTestsDefinitions::ROIAlignLayerTest;

namespace {
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inputShapes = {{2, 3, 16, 16}};
const std::vector<std::vector<std::vector<float>>> rois = {{{0.1, 0.2, 0.44, 0.34}, {0.33, 0.41, 0.11, 0.32}}};
const std::vector<std::vector<size_t>> batchIndices = {{0, 1}};
const std::vector<size_t> numROIs = {2};
const std::vector<size_t> pooledHs = {3};
const std::vector<size_t> pooledWs = {3};
const std::vector<size_t> samplingRatios = {2, 0};
const std::vector<float> spatialScale = {16.0};
const std::vector<std::string> modes = {"avg", "max"};

const auto roiAlignArgSet = ::testing::Combine(::testing::ValuesIn(inputShapes),     //
                                               ::testing::ValuesIn(rois),            //
                                               ::testing::ValuesIn(batchIndices),    //
                                               ::testing::ValuesIn(numROIs),         //
                                               ::testing::ValuesIn(pooledHs),        //
                                               ::testing::ValuesIn(pooledWs),        //
                                               ::testing::ValuesIn(samplingRatios),  //
                                               ::testing::ValuesIn(spatialScale),    //
                                               ::testing::ValuesIn(modes)            //
);

INSTANTIATE_TEST_CASE_P(ROIAlign, ROIAlignLayerTest,
                        ::testing::Combine(roiAlignArgSet,                                       //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        ROIAlignLayerTest::getTestCaseName);
}  // namespace
