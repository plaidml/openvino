// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>
#include "transformations/utils/pass_param.hpp"

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertShuffleChannels3;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertShuffleChannels3: public ngraph::pass::GraphRewrite, public ngraph::pass::PassParam {
public:
    ConvertShuffleChannels3() : GraphRewrite(), PassParam() {
        convert_shuffle_channels3();
    }

private:
    void convert_shuffle_channels3();
};
