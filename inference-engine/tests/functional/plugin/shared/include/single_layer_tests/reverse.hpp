// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/reverse.hpp"

namespace LayerTestsDefinitions {

TEST_P(ReverseLayerTest, CompareWithRefsDynamicBath) {
    Run();
};

} // namespace LayerTestsDefinitions
