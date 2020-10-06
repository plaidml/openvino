// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpu_layers_tests.hpp"

#include <vpu/middleend/pass_manager.hpp>
#include <vpu/frontend/frontend.hpp>
#include <cpp_interfaces/base/ie_inference_plugin_api.hpp>

class graphTransformerFunctionalTests : public vpuLayersTests {
protected:
    void SetUp() override;

    void CreateModel();
    void PrepareGraphCompilation();
    void InitializeInputData(const vpu::DataDesc& inputDataDesc);

    vpu::Data InitializeOutputData(const vpu::DataDesc& outputDataDesc);

    /// @returns execution time in microseconds
    int64_t CompileAndInfer(InferenceEngine::Blob::Ptr& inputBlob,
                            InferenceEngine::Blob::Ptr& outputBlob,
                            bool lockLayout = false);

protected:
   vpu::ModelPtr          _gtModel;
   vpu::CompilationConfig _compilationConfig;
   vpu::StageBuilder::Ptr _stageBuilder;
   vpu::Data              _dataIntermediate;

private:
   vpu::Platform                      _platform = vpu::Platform::MYRIAD_X;
   vpu::FrontEnd::Ptr                 _frontEnd;
   vpu::PassManager::Ptr              _passManager;
   vpu::BackEnd::Ptr                  _backEnd;
   InferenceEngine::ExecutableNetwork _executableNetwork;
};
