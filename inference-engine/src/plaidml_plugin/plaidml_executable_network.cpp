// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// NB: IE_SET_METRIC requires this header to be included first
#include "ie_metric_helpers.hpp"

#include "plaidml_executable_network.hpp"

#include <vector>

#include "details/ie_cnn_network_tools.h"

#include "plaidml/op/op.h"

#include "plaidml_infer_request.hpp"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

InferRequestInternal::Ptr PlaidMLExecutableNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                           OutputsDataMap networkOutputs) {
  std::vector<edsl::Tensor> inputs;
  for (const auto& kvp : networkInputs) {
    inputs.push_back(tensorIONameMap_.at(kvp.first));
  }
  std::vector<edsl::Tensor> outputs;
  for (const auto& kvp : networkOutputs) {
    outputs.push_back(tensorIONameMap_.at(kvp.first));
  }
  Program program = edsl::buildProgram("ie", inputs, outputs);
  program.compile();
  return std::make_shared<PlaidMLInferRequest>(networkInputs, networkOutputs, program);
}

PlaidMLExecutableNetwork::PlaidMLExecutableNetwork(const ICNNNetwork& network, const std::string& device) {
  auto fcn = network.getFunction();
  IE_ASSERT(fcn);  // PlaidML requires that the nGraph-based API be used
  for (const std::shared_ptr<ngraph::Node>& node : fcn->get_ordered_ops()) {
    // TODO: Clean up how these cases are selected
    if (node->is_constant() || node->description() == "Constant") {
      handleConstant(node);
    } else if (node->is_parameter() || node->description() == "Parameter") {
      handleParameter(node);
    } else if (node->is_output() || node->description() == "Result") {
      handleOutput(node);
    } else {
      handleOp(node);
    }
  }
}

void PlaidMLExecutableNetwork::handleConstant(const std::shared_ptr<ngraph::Node>& node) {
  IE_ASSERT(node->get_output_size() == 1);
  IE_ASSERT(node->description() == "Constant");
  auto type = to_plaidml(node->get_element_type());
  std::vector<int64_t> dims{node->get_shape().begin(), node->get_shape().end()};
  TensorShape shape(type, dims);
  Buffer buffer(shape);
  // Specially resolve the constant-creating op
  Context ctx{node.get()};
  auto* layer = dynamic_cast<ngraph::opset1::Constant*>(ctx.layer);
  buffer.copy_from(layer->get_data_ptr());
  auto tensor = edsl::Constant(buffer, node->get_friendly_name());
  tensorMap_[std::make_pair(node->get_name(), 0)] = tensor;
}

void PlaidMLExecutableNetwork::handleParameter(const std::shared_ptr<ngraph::Node>& node) {
  IE_ASSERT(node->get_output_size() == 1);
  std::vector<int64_t> dims{node->get_shape().begin(), node->get_shape().end()};
  auto type = to_plaidml(node->get_element_type());
  auto tensor = edsl::Placeholder(type, dims, node->get_friendly_name());
  tensorMap_[std::make_pair(node->get_name(), 0)] = tensor;
  tensorIONameMap_[node->get_friendly_name()] = tensor;
}

void PlaidMLExecutableNetwork::handleOutput(const std::shared_ptr<ngraph::Node>& node) {
  // The OV output name is the name of the node _prior_ to the result
  const auto& src_output = node->input(0).get_source_output();
  tensorIONameMap_[src_output.get_node()->get_friendly_name()] =
      tensorMap_.at(std::make_pair(src_output.get_node()->get_name(), src_output.get_index()));
}

void PlaidMLExecutableNetwork::handleOp(const std::shared_ptr<ngraph::Node>& node) {
  auto op = OpsRegistry::instance()->resolve(node->description());
  if (!op) {
    THROW_IE_EXCEPTION << "Unsupported operation: " << node->description();
  }

  Context ctx{node.get()};
  for (const auto& input : node->inputs()) {
    const auto& src_output = input.get_source_output();
    const auto& name = src_output.get_node()->get_name();
    const auto& index = src_output.get_index();
    auto tensor = tensorMap_.at(std::make_pair(name, index));
    ctx.operands.push_back(tensor);
  }
  auto value = op(ctx);
  auto tuple = value.as_tuple();
  IE_ASSERT(tuple.size() == node->get_output_size());
  for (unsigned i = 0; i < tuple.size(); i++) {
    auto tensor = tuple.at(i).as_tensor();
    tensorMap_[std::make_pair(node->get_name(), i)] = tensor;
  }
}

void PlaidMLExecutableNetwork::GetMetric(const std::string& name, Parameter& result, ResponseDesc* resp) const {
  if (name == METRIC_KEY(SUPPORTED_METRICS)) {
    std::vector<std::string> metrics = {
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
    };
    result = IE_SET_METRIC(SUPPORTED_METRICS, metrics);
  } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
    result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, 1);
  } else {
    THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork metric: " << name;
  }
}

}  // namespace PlaidMLPlugin
