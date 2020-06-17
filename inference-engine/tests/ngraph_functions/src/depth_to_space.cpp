// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ngraph::Node> makeDepthToSpace(const ngraph::Output<Node> &in,
                                               ngraph::opset1::DepthToSpace::DepthToSpaceMode mode,
                                               size_t blockSize) {
    auto dtsNode = std::make_shared<ngraph::opset1::DepthToSpace>(in, mode, blockSize);
    return dtsNode;
}

}  // namespace builder
}  // namespace ngraph
