// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API ConvertDivide;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertDivide: public ngraph::pass::GraphRewrite {
public:
    ConvertDivide() : GraphRewrite() {
        convert_divide();
    }

private:
    void convert_divide();
};
