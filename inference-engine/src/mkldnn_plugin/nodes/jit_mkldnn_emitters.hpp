// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common/emitter.h"
#include <cpu/x64/jit_generator.hpp>
#include "mkldnn_node.h"
#include <cpu/x64/jit_uni_eltwise_injector.hpp>


namespace MKLDNNPlugin {

class jit_mkldnn_emitter : public jit_emitter {
public:
    jit_mkldnn_emitter(mkldnn::impl::cpu::x64::jit_generator *host, mkldnn::impl::cpu::x64::cpu_isa_t host_isa, const MKLDNNNode* node,
                       InferenceEngine::Precision exec_prc = InferenceEngine::Precision::FP32);

    size_t get_inputs_num() override;

    void emit(const std::vector<size_t> &in_vec_idxs, const std::vector<size_t> &out_vec_idxs,
              const std::vector<size_t> &pool_vec_idxs, const std::vector<size_t> &pool_gpr_idxs) override;

    void emit_table() override;

private:
    std::shared_ptr<mkldnn::impl::cpu::x64::jit_uni_eltwise_injector_f32<mkldnn::impl::cpu::x64::sse41>> eltwise_injector_sse42;
    std::shared_ptr<mkldnn::impl::cpu::x64::jit_uni_eltwise_injector_f32<mkldnn::impl::cpu::x64::avx2>> eltwise_injector_avx2;
    std::shared_ptr<mkldnn::impl::cpu::x64::jit_uni_eltwise_injector_f32<mkldnn::impl::cpu::x64::avx512_common>> eltwise_injector_avx512_common;
};

} // namespace MKLDNNPlugin
