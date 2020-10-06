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

class TRANSFORMATIONS_API ConvertTileToIETile;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertTileToIETile: public ngraph::pass::GraphRewrite {
public:
    ConvertTileToIETile() : GraphRewrite() {
        convert_tile();
    }

private:
    void convert_tile();
};
