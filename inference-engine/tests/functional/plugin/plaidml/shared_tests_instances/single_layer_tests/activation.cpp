// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/activation.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

using LayerTestsDefinitions::ActivationLayerTest;
using ngraph::helpers::ActivationTypes;
// using ngraph::helpers::Abs;
// using ngraph::helpers::Acos;
// using ngraph::helpers::Asin;
// using ngraph::helpers::Atan;
// using ngraph::helpers::Ceiling;
// using ngraph::helpers::Clamp;
// using ngraph::helpers::Cos;
// using ngraph::helpers::Cosh;
// using ngraph::helpers::Elu;
// using ngraph::helpers::Erf;
// using ngraph::helpers::Exp;
// using ngraph::helpers::Floor;
// // using ngraph::helpers::Gelu;  // not in opset1
// using ngraph::helpers::HardSigmoid;
// // using ngraph::op::LeakyRelu;
// using ngraph::helpers::Log;
// // using ngraph::helpers::Mish;  // not in opset1
// using ngraph::helpers::Negative;
// // using ngraph::op::PReLu;
// using ngraph::helpers::Relu;
// using ngraph::helpers::Selu;
// using ngraph::helpers::Sigmoid;
// using ngraph::helpers::Sign;
// using ngraph::helpers::Sin;
// using ngraph::helpers::Sinh;
// using ngraph::helpers::Sqrt;
// using ngraph::helpers::Tan;
// using ngraph::helpers::Tanh;

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

// TODO Below is old

// TODO: Missing PReLU types, see
// inference-engine/tests/functional/plugin/cpu/shared_tests_instances/single_layer_tests/activation.cpp

// TODO: Missing PReLU types, see
// inference-engine/tests/functional/plugin/cpu/shared_tests_instances/single_layer_tests/activation.cpp

// std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
//     {{1, 50}, {{}}},
//     {{1, 128}, {{}}},
// };

// const auto basicCases = ::testing::Combine(::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),  //
//                                            ::testing::ValuesIn(inputPrecisions),                                    //
//                                            ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),            //
//                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

// INSTANTIATE_TEST_SUITE_P(Activation_Basic, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);

// }  // namespace
