// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/activation.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

using LayerTestsDefinitions::ActivationLayerTest;
using ngraph::helpers::ActivationTypes;

namespace {
// Common params
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
        // InferenceEngine::Precision::FP16
};

const std::vector<ActivationTypes> activationTypes = {
        ActivationTypes::Sigmoid,
        ActivationTypes::Tanh,
        ActivationTypes::Relu,
        ActivationTypes::Exp,
        ActivationTypes::Log,
        ActivationTypes::Sign,
        ActivationTypes::Abs,
        // ActivationTypes::Clamp,
        // ActivationTypes::Negative,
        // ActivationTypes::Acos,
        // ActivationTypes::Asin,
        // ActivationTypes::Atan,
        // ActivationTypes::Cos,
        // ActivationTypes::Cosh,
        // ActivationTypes::Floor,
        // ActivationTypes::Sin,
        // ActivationTypes::Sinh,
        // ActivationTypes::Sqrt,
        // ActivationTypes::Tan,
        // ActivationTypes::Elu,
        // ActivationTypes::Erf,
        // ActivationTypes::HardSigmoid,
        // ActivationTypes::Selu,
        // ActivationTypes::Ceiling,
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(activationTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(std::vector<size_t >({1, 50}), std::vector<size_t >({1, 128})),
        ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)
);


INSTANTIATE_TEST_CASE_P(Activation_Basic, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);

}  // namespace
