// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <paddlepaddle_frontend/place.hpp>
#include "decoder.hpp"
#include "framework.pb.h"

using namespace ngraph;
using namespace frontend;

bool PlacePDPD::is_input() const
{
    const auto& model_ins = m_input_model.get_inputs();

    const auto cmp = [this](const Place::Ptr& p) { return p.get() == this; };
    return std::find_if(model_ins.begin(), model_ins.end(), cmp) != model_ins.end();
}

bool PlacePDPD::is_output() const
{
    const auto& model_outs = m_input_model.get_outputs();
    const auto cmp = [this](const Place::Ptr& p) { return p.get() == this; };
    return std::find_if(model_outs.begin(), model_outs.end(), cmp) != model_outs.end();
}

OpPlacePDPD::OpPlacePDPD(const InputModel& input_model,
                         const std::vector<std::string>& names,
                         const std::shared_ptr<paddle::framework::proto::OpDesc>& op_desc)
    : PlacePDPD(input_model, names)
    , m_op_desc(op_desc)
{
}

OpPlacePDPD::OpPlacePDPD(const InputModel& input_model,
                         const std::shared_ptr<paddle::framework::proto::OpDesc>& op_desc)
    : OpPlacePDPD(input_model, {}, op_desc)
{
}

TensorPlacePDPD::TensorPlacePDPD(const InputModel& input_model,
                                 const std::vector<std::string>& names,
                                 const std::shared_ptr<paddle::framework::proto::VarDesc>& var_desc)
    : PlacePDPD(input_model, names)
    , m_var_desc(var_desc)
{
    const auto& var_type = var_desc->type();
    if (var_type.type() == paddle::framework::proto::VarType::LOD_TENSOR)
    {
        const auto& tensor_desc = var_type.lod_tensor().tensor();
        m_type = TYPE_MAP[tensor_desc.data_type()];
        m_pshape = PartialShape(
            std::vector<Dimension>(tensor_desc.dims().begin(), tensor_desc.dims().end()));
    }
}

TensorPlacePDPD::TensorPlacePDPD(const InputModel& input_model,
                                 const std::shared_ptr<paddle::framework::proto::VarDesc>& var_desc)
    : TensorPlacePDPD(input_model, {var_desc->name()}, var_desc)
{
}

std::vector<Place::Ptr> TensorPlacePDPD::get_consuming_ports() const
{
    std::vector<Place::Ptr> consuming_ports;
    for (const auto& consuming_port : m_consuming_ports)
    {
        if (const auto& locked = consuming_port.lock())
        {
            consuming_ports.push_back(locked);
        }
        else
        {
            FRONT_END_THROW("Consuming Port has expired.");
        }
    }
    return consuming_ports;
}

Place::Ptr TensorPlacePDPD::get_producing_port() const
{
    FRONT_END_GENERAL_CHECK(m_producing_ports.size() > 1, "Only one producing port is supported.");
    if (const auto& producing_port = m_producing_ports[0].lock())
    {
        return producing_port;
    }
    FRONT_END_THROW("Producing Port has expired.");
}

std::shared_ptr<TensorPlacePDPD> InPortPlacePDPD::getSourceTensorPDPD() const
{
    if (const auto& tensor = m_source_tensor.lock())
    {
        return tensor;
    }
    FRONT_END_THROW("Source Tensor has expired.");
}

std::shared_ptr<OpPlacePDPD> InPortPlacePDPD::getOp()
{
    if (const auto& op = m_op.lock())
    {
        return op;
    }
    FRONT_END_THROW("Operation has expired.");
}

std::shared_ptr<TensorPlacePDPD> OutPortPlacePDPD::getTargetTensorPDPD() const
{
    if (const auto& target_tensor = m_target_tensor.lock())
    {
        return target_tensor;
    }
    FRONT_END_THROW("Target Tensor has expired.");
}
