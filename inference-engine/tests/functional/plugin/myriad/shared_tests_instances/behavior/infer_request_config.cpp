// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "multi-device/multi_device_config.hpp"
#include "vpu/vpu_plugin_config.hpp"
#include "behavior/infer_request_config.hpp"

namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    const std::vector<std::map<std::string, std::string>> multiConfigs = {
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , CommonTestUtils::DEVICE_MYRIAD}}
    };

    const std::vector<std::map<std::string, std::string>> Inconfigs = {
            {},
            {{VPU_CONFIG_KEY(IGNORE_IR_STATISTIC), CONFIG_VALUE(YES)}},
            {{VPU_CONFIG_KEY(IGNORE_IR_STATISTIC), CONFIG_VALUE(NO)}},

            {{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(YES)}},
            {{VPU_MYRIAD_CONFIG_KEY(FORCE_RESET), CONFIG_VALUE(NO)}},

            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_NONE)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_ERROR)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_WARNING)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_INFO)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_DEBUG)}},
            {{CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_TRACE)}},

            {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)}},
            {{VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(NO)}},

            {{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(YES)}},
            {{VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME), CONFIG_VALUE(NO)}}
    };

    const std::vector<std::map<std::string, std::string>> InmultiConfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {CONFIG_KEY(LOG_LEVEL), CONFIG_VALUE(LOG_DEBUG)}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_MYRIAD},
            {VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION), CONFIG_VALUE(YES)}}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(configs)),
                            InferConfigTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, InferConfigTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiConfigs)),
                            InferConfigTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferConfigInTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                    ::testing::ValuesIn(Inconfigs)),
                            InferConfigInTests::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(smoke_Multi_BehaviorTests, InferConfigInTests,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_MULTI),
                                    ::testing::ValuesIn(InmultiConfigs)),
                            InferConfigInTests::getTestCaseName);
}  // namespace
