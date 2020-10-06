﻿// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <algorithm>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "details/ie_cnn_network_tools.h"

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/unicode_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/test_model/test_model.hpp"
#include "functional_test_utils/network_utils.hpp"


#ifdef ENABLE_UNICODE_PATH_SUPPORT

#include <iostream>

#define GTEST_COUT std::cerr << "[          ] [ INFO ] "

#include <codecvt>

#endif

using NetReaderNoParamTest = CommonTestUtils::TestsCommon;

TEST_F(NetReaderNoParamTest, IncorrectModel) {
    InferenceEngine::Core ie;
    ASSERT_THROW(ie.ReadNetwork("incorrectFilePath"), InferenceEngine::details::InferenceEngineException);
}

using NetReaderTestParams = std::tuple<InferenceEngine::SizeVector, InferenceEngine::Precision>;

class NetReaderTest
        : public CommonTestUtils::TestsCommon, public testing::WithParamInterface<NetReaderTestParams> {
protected:
    static void read(const std::string &modelPath, const std::string &weightsPath, InferenceEngine::Core &ie,
                     InferenceEngine::CNNNetwork &network) {
        network = ie.ReadNetwork(modelPath, weightsPath);
    }

    void SetUp() override {
        std::tie(_inputDims, _netPrc) = GetParam();
        (void) FuncTestUtils::TestModel::generateTestModel(_modelPath,
                                                             _weightsPath,
                                                             _netPrc,
                                                             _inputDims,
                                                             &_refLayers);
    }

    void TearDown() override {
        CommonTestUtils::removeIRFiles(_modelPath, _weightsPath);
    }

    /* validates a read network with the reference map of CNN layers */
    void compareWithRef(const InferenceEngine::CNNNetwork &network,
                        const std::vector<InferenceEngine::CNNLayerPtr> &refLayersVec) {
        ASSERT_NO_THROW(FuncTestUtils::compareLayerByLayer<std::vector<InferenceEngine::CNNLayerPtr>>(
                InferenceEngine::details::CNNNetSortTopologically(network),
                refLayersVec, false));
    }

    const std::string _modelPath = "NetReader_test.xml";
    const std::string _weightsPath = "NetReader_test.bin";
    InferenceEngine::SizeVector _inputDims;
    InferenceEngine::Precision _netPrc;

    std::vector<InferenceEngine::CNNLayerPtr> _refLayers;
};

TEST_P(NetReaderTest, ReadCorrectModelWithWeightsAndValidate) {
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork network;
    read(_modelPath, _weightsPath, ie, network);

    for (auto input : network.getInputsInfo()) {
        input.second->setPrecision(_netPrc);
    }
    for (auto input : network.getOutputsInfo()) {
        input.second->setPrecision(_netPrc);
    }

    compareWithRef(network, _refLayers);
}

TEST_P(NetReaderTest, ReadNetworkTwiceSeparately) {
    InferenceEngine::Core ie;

    InferenceEngine::CNNNetwork network;
    read(_modelPath, _weightsPath, ie, network);

    InferenceEngine::CNNNetwork network2;
    read(_modelPath, _weightsPath, ie, network2);

    auto& icnn = static_cast<InferenceEngine::ICNNNetwork &>(network);
    auto& icnn2 = static_cast<InferenceEngine::ICNNNetwork &>(network2);

    ASSERT_NE(&icnn,
              &icnn2);
    ASSERT_NO_THROW(FuncTestUtils::compareCNNNetworks(network, network2));
}

#ifdef ENABLE_UNICODE_PATH_SUPPORT

TEST_P(NetReaderTest, ReadCorrectModelWithWeightsUnicodePath) {
    GTEST_COUT << "params.modelPath: '" << _modelPath << "'" << std::endl;
    GTEST_COUT << "params.weightsPath: '" << _weightsPath << "'" << std::endl;
    GTEST_COUT << "params.netPrc: '" << _netPrc.name() << "'" << std::endl;
    for (std::size_t testIndex = 0; testIndex < CommonTestUtils::test_unicode_postfix_vector.size(); testIndex++) {
        std::wstring postfix = L"_" + CommonTestUtils::test_unicode_postfix_vector[testIndex];
        std::wstring modelPath = CommonTestUtils::addUnicodePostfixToPath(_modelPath, postfix);
        std::wstring weightsPath = CommonTestUtils::addUnicodePostfixToPath(_weightsPath, postfix);
        try {
            bool is_copy_successfully;
            is_copy_successfully = CommonTestUtils::copyFile(_modelPath, modelPath);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << _modelPath << "' to '"
                       << InferenceEngine::details::wStringtoMBCSstringChar(modelPath) << "'";
            }
            is_copy_successfully = CommonTestUtils::copyFile(_weightsPath, weightsPath);
            if (!is_copy_successfully) {
                FAIL() << "Unable to copy from '" << _weightsPath << "' to '"
                       << InferenceEngine::details::wStringtoMBCSstringChar(weightsPath) << "'";
            }
            GTEST_COUT << "Test " << testIndex << std::endl;
            InferenceEngine::Core ie;
            ASSERT_NO_THROW(ie.ReadNetwork(modelPath, weightsPath));
            CommonTestUtils::removeFile(modelPath);
            CommonTestUtils::removeFile(weightsPath);
            GTEST_COUT << "OK" << std::endl;
        }
        catch (const InferenceEngine::details::InferenceEngineException &e_next) {
            CommonTestUtils::removeFile(modelPath);
            CommonTestUtils::removeFile(weightsPath);
            FAIL() << e_next.what();
        }
    }
}

#endif

std::string getTestCaseName(testing::TestParamInfo<NetReaderTestParams> testParams) {
    InferenceEngine::SizeVector dims;
    InferenceEngine::Precision prc;
    std::tie(dims, prc) = testParams.param;

    std::ostringstream ss;
    std::copy(dims.begin(), dims.end()-1, std::ostream_iterator<size_t>(ss, "_"));
    ss << dims.back() << "}_" << prc.name();
    return "{" + ss.str();
}

static const auto params = testing::Combine(
        testing::Values(InferenceEngine::SizeVector{1, 3, 227, 227}),
        testing::Values(InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16));

INSTANTIATE_TEST_CASE_P(
        NetReaderTest,
        NetReaderTest,
        params,
        getTestCaseName);

#ifdef GTEST_COUT
#undef GTEST_COUT
#endif
