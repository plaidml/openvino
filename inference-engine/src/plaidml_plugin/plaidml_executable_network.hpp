// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"

#include "ngraph/descriptor/tensor.hpp"

#include "ngraph/function.hpp"  // TODO

#include "plaidml/edsl/edsl.h"

namespace PlaidMLPlugin {

struct TODOnGraphParts {
  std::shared_ptr<const ngraph::Function> fcn;
  InferenceEngine::InputsDataMap networkInputs;
  InferenceEngine::OutputsDataMap networkOutputs;
  TODOnGraphParts(const InferenceEngine::ICNNNetwork& network) : fcn(network.getFunction()) {
    network.getInputsInfo(networkInputs);
    network.getOutputsInfo(networkOutputs);
  }
};

class PlaidMLExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
 public:
  using Ptr = std::shared_ptr<PlaidMLExecutableNetwork>;

  PlaidMLExecutableNetwork(const InferenceEngine::ICNNNetwork& network, const std::string& device);
  virtual ~PlaidMLExecutableNetwork() = default;

  InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(
      InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs) override;

  void GetMetric(const std::string& name, InferenceEngine::Parameter& result,
                 InferenceEngine::ResponseDesc* resp) const override;

 private:
  void handleConstant(const std::shared_ptr<ngraph::Node>& node);
  void handleParameter(const std::shared_ptr<ngraph::Node>& node);
  void handleOutput(const std::shared_ptr<ngraph::Node>& node);
  void handleOp(const std::shared_ptr<ngraph::Node>& node);

  plaidml::Program TODOMakeProgram(
      std::shared_ptr<const ngraph::Function> fcn,
      InferenceEngine::InputsDataMap networkInputs,
      InferenceEngine::OutputsDataMap networkOutputs);

 private:
  // Lets us look up the PlaidML tensor by the name of the node that produces it and the index of which output it is
  std::map<std::pair<std::string, size_t>, plaidml::edsl::Tensor> tensorMap_;

  // Go from the names OV uses for a networks inputs and outputs to the corresponding PlaidML Tensor
  std::map<std::string, plaidml::edsl::Tensor> tensorIONameMap_;

  // The inputs & outputs maps & Function, read from the nGraph graph at construction
  TODOnGraphParts todo_parts;

  plaidml::Program program_;
};

}  // namespace PlaidMLPlugin
