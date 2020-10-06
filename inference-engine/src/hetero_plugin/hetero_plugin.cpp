// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "hetero_plugin.hpp"
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <fstream>
#include <unordered_set>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include "hetero_executable_network.hpp"
#include <ie_util_internal.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::HeteroConfigParams;
using namespace HeteroPlugin;
using namespace std;

static Version heteroPluginDescription = {
        {2, 1},  // plugin API version
        CI_BUILD_NUMBER,
        "heteroPlugin"  // plugin description message
};

Engine::Engine() {
    _pluginName = "HETERO";
    _config[KEY_EXCLUSIVE_ASYNC_REQUESTS] = YES;
    _config[HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)] = NO;
}

namespace {

Engine::Configs mergeConfigs(Engine::Configs config, const Engine::Configs & local) {
    for (auto && kvp : local) {
        config[kvp.first] = kvp.second;
    }
    return config;
}

}  // namespace

InferenceEngine::ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const InferenceEngine::ICNNNetwork&    network,
                                                                           const Configs&                   config) {
    if (GetCore() == nullptr) {
        THROW_IE_EXCEPTION << "Please, work with HETERO device via InferencEngine::Core object";
    }

    return std::make_shared<HeteroExecutableNetwork>(*cloneNet(network), mergeConfigs(_config, config), this);
}

ExecutableNetwork Engine::ImportNetworkImpl(std::istream& heteroModel, const Configs& config) {
    if (GetCore() == nullptr) {
        THROW_IE_EXCEPTION << "Please, work with HETERO device via InferencEngine::Core object";
    }

    IExecutableNetwork::Ptr executableNetwork;
    executableNetwork.reset(new ExecutableNetworkBase<ExecutableNetworkInternal>(
                                std::make_shared<HeteroExecutableNetwork>(heteroModel, mergeConfigs(_config, config), this)),
                            [](InferenceEngine::details::IRelease *p) {p->Release();});

    return ExecutableNetwork{executableNetwork};
}

Engine::Configs Engine::GetSupportedConfig(const Engine::Configs& config, const std::string & deviceName) const {
    std::vector<std::string> supportedConfigKeys = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    Engine::Configs supportedConfig;
    for (auto&& key : supportedConfigKeys) {
        auto itKey = config.find(key);
        if (config.end() != itKey) {
            supportedConfig[key] = itKey->second;
        }
    }
    return supportedConfig;
}

Engine::DeviceMetaInformationMap Engine::GetDevicePlugins(const std::string& targetFallback,
                                                          const Configs & localConfig) const {
    auto getDeviceConfig = [&](const std::string & deviceWithID) {
        DeviceIDParser deviceParser(deviceWithID);
        std::string deviceName = deviceParser.getDeviceName();
        Configs tconfig = mergeConfigs(_config, localConfig);

        // set device ID if any
        std::string deviceIDLocal = deviceParser.getDeviceID();
        if (!deviceIDLocal.empty()) {
            tconfig[KEY_DEVICE_ID] = deviceIDLocal;
        }

        return GetSupportedConfig(tconfig, deviceName);
    };

    auto fallbackDevices = InferenceEngine::DeviceIDParser::getHeteroDevices(targetFallback);
    Engine::DeviceMetaInformationMap metaDevices;
    for (auto&& deviceName : fallbackDevices) {
        auto itPlugin = metaDevices.find(deviceName);
        if (metaDevices.end() == itPlugin) {
            metaDevices[deviceName] = getDeviceConfig(deviceName);
        }
    }
    return metaDevices;
}

void Engine::SetConfig(const Configs &configs) {
    for (auto&& config : configs) {
        _config[config.first] = config.second;
    }
}

HeteroLayerColorer::HeteroLayerColorer(const std::vector<std::string>& devices) {
    static const std::vector<std::string> colors = {"#5A5DF0", "#20F608", "#F1F290", "#11F110"};
    for (auto&& device : devices) {
        deviceColorMap[device] = colors[std::distance(&device, devices.data()) % colors.size()];
    }
}

void HeteroLayerColorer::operator()(const CNNLayerPtr layer,
                ordered_properties &printed_properties,
                ordered_properties &node_properties) {
    auto device = layer->affinity;
    printed_properties.insert(printed_properties.begin(), std::make_pair("device", device));
    node_properties.emplace_back("fillcolor", deviceColorMap[device]);
}

void Engine::SetAffinity(InferenceEngine::ICNNNetwork &network, const Configs &config) {
    QueryNetworkResult qr;
    QueryNetwork(network, config, qr);

    details::CNNNetworkIterator i(&network);
    while (i != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *i;
        auto it = qr.supportedLayersMap.find(layer->name);
        if (it != qr.supportedLayersMap.end()) {
            layer->affinity = it->second;
        }
        i++;
    }

    auto dumpDot = [](const Configs & config) {
        auto it = config.find(HETERO_CONFIG_KEY(DUMP_GRAPH_DOT));
        return it != config.end() ? it->second == YES : false;
    };

    if (dumpDot(config) || dumpDot(_config)) {
        std::unordered_set<std::string> devicesSet;
        details::CNNNetworkIterator i(&network);
        while (i != details::CNNNetworkIterator()) {
            CNNLayer::Ptr layer = *i;
            if (!layer->affinity.empty()) {
                devicesSet.insert(layer->affinity);
            }
            i++;
        }
        std::vector<std::string> devices{std::begin(devicesSet), std::end(devicesSet)};
        std::stringstream stream(std::stringstream::out);
        stream << "hetero_affinity_" << network.getName() << ".dot";

        std::ofstream file(stream.str());
        saveGraphToDot(network, file, HeteroLayerColorer{devices});
    }
}

void Engine::QueryNetwork(const ICNNNetwork &network, const Configs& config, QueryNetworkResult &qr) const {
    if (GetCore() == nullptr) {
        THROW_IE_EXCEPTION << "Please, work with HETERO device via InferencEngine::Core object";
    }

    auto tconfig = mergeConfigs(_config, config);
    auto it = tconfig.find("TARGET_FALLBACK");
    if (it == tconfig.end()) {
        THROW_IE_EXCEPTION << "The 'TARGET_FALLBACK' option was not defined for heterogeneous plugin";
    }

    std::string fallbackDevicesStr = it->second;
    DeviceMetaInformationMap metaDevices = GetDevicePlugins(fallbackDevicesStr, tconfig);

    std::map<std::string, QueryNetworkResult> queryResults;
    // go over devices and call query network
    for (auto&& metaDevice : metaDevices) {
        auto& deviceName = metaDevice.first;
        queryResults[deviceName] = GetCore()->QueryNetwork(network, deviceName, metaDevice.second);
    }

    //  WARNING: Here is devices with user set priority
    auto fallbackDevices = InferenceEngine::DeviceIDParser::getHeteroDevices(fallbackDevicesStr);

    details::CNNNetworkIterator i(&network);
    while (i != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *i;
        for (auto&& deviceName : fallbackDevices) {
            auto& deviceQueryResult = queryResults[deviceName];
            if (deviceQueryResult.supportedLayersMap.find(layer->name) != deviceQueryResult.supportedLayersMap.end()) {
                qr.supportedLayersMap[layer->name] = deviceName;
                break;
            }
        }
        i++;
    }

    // set OK status
    qr.rc = StatusCode::OK;
}

Parameter Engine::GetMetric(const std::string& name, const std::map<std::string, Parameter> & /*options*/) const {
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, std::vector<std::string>{
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(FULL_DEVICE_NAME),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)});
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{
            HETERO_CONFIG_KEY(DUMP_GRAPH_DOT),
            "TARGET_FALLBACK",
            CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)});
    } else if (METRIC_KEY(FULL_DEVICE_NAME) == name) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, std::string{"HETERO"});
    } else {
        THROW_IE_EXCEPTION << "Unsupported Plugin metric: " << name;
    }
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter> & /*options*/) const {
    if (name == HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)) {
        auto it = _config.find(HETERO_CONFIG_KEY(DUMP_GRAPH_DOT));
        IE_ASSERT(it != _config.end());
        bool dump = it->second == YES;
        return { dump };
    } else if (name == "TARGET_FALLBACK") {
        auto it = _config.find("TARGET_FALLBACK");
        if (it == _config.end()) {
            THROW_IE_EXCEPTION << "Value for TARGET_FALLBACK is not set";
        } else {
            return { it->second };
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported config key: " << name;
    }
}

IE_SUPPRESS_DEPRECATED_START

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode) CreatePluginEngine(
        InferenceEngine::IInferencePlugin *&plugin,
        InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin({{2, 1}, CI_BUILD_NUMBER, "heteroPlugin"},
                                           std::make_shared<Engine>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}
