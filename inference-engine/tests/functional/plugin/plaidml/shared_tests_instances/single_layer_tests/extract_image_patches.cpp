// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/extract_image_patches.hpp"

using LayerTestsDefinitions::ExtractImagePatchesTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16  // TODO: Not yet working
};

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 3}, {5, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {2, 2}};
const std::vector<std::vector<size_t>> rates = {{1, 1}, {2, 2}};
const std::vector<ngraph::op::PadType> padTypes = {
    ngraph::op::PadType::EXPLICIT,  //
    ngraph::op::PadType::VALID      //
};

INSTANTIATE_TEST_CASE_P(ExtractImagePatches2DTestCheck, ExtractImagePatchesTest,
                         ::testing::Combine(::testing::Values(std::vector<size_t>({1, 1, 10, 10})),    //
                                            ::testing::ValuesIn(kernels),                                  //
                                            ::testing::ValuesIn(strides),                                  //
                                            ::testing::ValuesIn(rates),                                    //
                                            ::testing::ValuesIn(padTypes),                                 //
                                            ::testing::ValuesIn(netPrecisions),                          //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        ExtractImagePatchesTest::getTestCaseName);

//
///* ============= 3D Convolution ============= */
//const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}, {3, 5, 3}};
//const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0}, {0, 2, 0}};
//
//const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}, {1, 2, 1}};
//const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}, {1, 2, 1}};
//
//const auto conv3DParams_ExplicitPadding = ::testing::Combine(::testing::ValuesIn(kernels3d),                   //
//                                                             ::testing::ValuesIn(strides3d),                   //
//                                                             ::testing::ValuesIn(paddings3d),                  //
//                                                             ::testing::ValuesIn(paddings3d),                  //
//                                                             ::testing::ValuesIn(dilations3d),                 //
//                                                             ::testing::Values(5),                             //
//                                                             ::testing::Values(ngraph::op::PadType::EXPLICIT)  //
//);
//const auto conv3DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels3d),                        //
//                                                          ::testing::ValuesIn(strides3d),                        //
//                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  //
//                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  //
//                                                          ::testing::ValuesIn(dilations3d),                      //
//                                                          ::testing::Values(5),                                  //
//                                                          ::testing::Values(ngraph::op::PadType::VALID)          //
//);
//
//INSTANTIATE_TEST_CASE_P(ExtractImagePatches3DTestCheck, ExtractImagePatchesTest,
//                         ::testing::Combine(conv3DParams_ExplicitPadding,                                //
//                                            ::testing::ValuesIn(netPrecisions),                          //
//                                            ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),  //
//                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
//                        ExtractImagePatchesTest::getTestCaseName);

}  // namespace
