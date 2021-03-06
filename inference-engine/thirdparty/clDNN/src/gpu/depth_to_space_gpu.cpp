// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "depth_to_space_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "depth_to_space/depth_to_space_kernel_selector.h"
#include "depth_to_space/depth_to_space_kernel_ref.h"
#include "error_handler.h"
#include "common_types.h"

using namespace cldnn;

namespace cldnn {
namespace gpu {
struct depth_to_space_gpu : typed_primitive_gpu_impl<depth_to_space> {
    using parent = typed_primitive_gpu_impl<depth_to_space>;
    using parent::parent;

public:
    static primitive_impl* create(const depth_to_space_node& arg) {
        auto depth_to_space_params = get_default_params<kernel_selector::depth_to_space_params>(arg);
        auto depth_to_space_optional_params =
            get_default_optional_params<kernel_selector::depth_to_space_optional_params>(arg.get_program());

        depth_to_space_params.block_size = arg.get_primitive()->block_size;
        depth_to_space_params.mode = arg.get_primitive()->mode == depth_to_space_mode::blocks_first ? kernel_selector::depth_to_space_mode::BLOCKS_FIRST
                                                                                                    : kernel_selector::depth_to_space_mode::DEPTH_FIRST;

        auto& kernel_selector = kernel_selector::depth_to_space_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(depth_to_space_params, depth_to_space_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto depth_to_space = new depth_to_space_gpu(arg, best_kernels[0]);

        return depth_to_space;
    }
};

namespace detail {

attach_depth_to_space_gpu::attach_depth_to_space_gpu() {
    auto val_fw = depth_to_space_gpu::create;
    implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfyx), val_fw);
    implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfyx), val_fw);
    implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::bfzyx), val_fw);
    implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::bfzyx), val_fw);
    implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::b_fs_yx_fsv16), val_fw);
    implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::b_fs_yx_fsv16), val_fw);
    implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::u8, format::b_fs_yx_fsv16), val_fw);
    implementation_map<depth_to_space>::add(std::make_tuple(engine_types::ocl, data_types::i8, format::b_fs_yx_fsv16), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
