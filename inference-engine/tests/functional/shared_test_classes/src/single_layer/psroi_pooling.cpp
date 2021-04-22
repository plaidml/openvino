// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/psroi_pooling.hpp"

namespace LayerTestsDefinitions {

std::string PSROIPoolingLayerTest::getTestCaseName(testing::TestParamInfo<psroiParams> obj) {
    std::vector<size_t> inputShape;
    std::vector<size_t> coordsShape;
    size_t outputDim;
    size_t groupSize;
    float spatialScale;
    size_t spatialBinsX;
    size_t spatialBinsY;
    std::string mode;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::tie(inputShape, coordsShape, outputDim, groupSize, spatialScale, spatialBinsX, spatialBinsY, mode, netPrecision, targetDevice) = obj.param;

    std::ostringstream result;

    result << "in_shape=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "coord_shape=" << CommonTestUtils::vec2str(coordsShape) << "_";
    result << "out_dim=" << outputDim << "_";
    result << "group_size=" << groupSize << "_";
    result << "scale=" << spatialScale << "_";
    result << "bins_x=" << spatialBinsX << "_";
    result << "bins_y=" << spatialBinsY << "_";
    result << "mode=" << mode << "_";
    result << "prec=" << netPrecision.name() << "_";
    result << "dev=" << targetDevice;
    return result.str();
}

void PSROIPoolingLayerTest::GenerateCoords(const std::vector<size_t>& feat_map_shape, float* buffer, size_t size) {
    const int height = feat_map_shape[2] / spatialScale_;
    const int width = feat_map_shape[3] / spatialScale_;
    CommonTestUtils::fill_data_roi(buffer, size, feat_map_shape[0] - 1, height, width, 1.0f, true);
}

static int randInt(int low, int high) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(low, high);
    return dis(gen);
}

static void fillROITensor(float* buffer, int numROIs, int batchSize,
                          int height, int width, int groupSize,
                          float spatialScale, int spatialBinsX, int spatialBinsY, const std::string& mode) {
    int minRoiWidth = groupSize;
    int maxRoiWidth = width / groupSize * groupSize;
    int minRoiHeight = groupSize;
    int maxRoiHeight = height / groupSize * groupSize;
    float scaleX = spatialScale;
    float scaleY = spatialScale;
    if (mode == "bilinear") {
        minRoiWidth = spatialBinsX;
        maxRoiWidth = width / spatialBinsX * spatialBinsX;
        minRoiHeight = spatialBinsY;
        maxRoiHeight = height / spatialBinsY * spatialBinsY;
        scaleX *= width;
        scaleY *= height;
    }
    int batchId = 0;
    for (int i = 0; i < numROIs; i++) {
        int sizeX = std::min(width, randInt(minRoiWidth, maxRoiWidth));
        int sizeY = std::min(height, randInt(minRoiHeight, maxRoiHeight));
        int startX = randInt(0, std::max(1, width - sizeX - 1));
        int startY = randInt(0, std::max(1, height - sizeY - 1));

        float* roi = buffer + i * 5;
        roi[0] = batchId;
        roi[1] = startX / scaleX;
        roi[2] = startY / scaleY;
        roi[3] = (startX + sizeX - 1) / scaleX;
        roi[4] = (startY + sizeY - 1) / scaleY;

        batchId = (batchId + 1) % batchSize;
    }
}

void PSROIPoolingLayerTest::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();

    auto inputShape = cnnNetwork.getInputShapes().begin()->second;

    size_t it = 0;
    for (const auto &input : cnnNetwork.getInputsInfo()) {
        const auto &info = input.second;
        InferenceEngine::Blob::Ptr blob;

        if (it == 1) {
            blob = make_blob_with_precision(info->getTensorDesc());
            blob->allocate();
            fillROITensor(blob->buffer(), blob->size() / 5,
                          inputShape[0], inputShape[2], inputShape[3], groupSize_,
                          spatialScale_, spatialBinsX_, spatialBinsY_, mode_);
        } else {
            blob = GenerateInput(*info);
        }
        inferRequest.SetBlob(info->name(), blob);
        inputs.push_back(blob);
        it++;
    }
    inferRequest.Infer();
}

void PSROIPoolingLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    std::vector<size_t> coordsShape;
    size_t outputDim;
    InferenceEngine::Precision netPrecision;
    std::tie(inputShape, coordsShape, outputDim, groupSize_, spatialScale_,
             spatialBinsX_, spatialBinsY_, mode_, netPrecision, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    size_t size = 1;
    for (auto dim : coordsShape) {
        size *= dim;
    }
    std::vector<float> coordsValue(size);
    GenerateCoords(inputShape, coordsValue.data(), size);
    auto constNode = ngraph::builder::makeConstant(ngPrc, coordsShape, coordsValue, false);
    std::shared_ptr<ngraph::Node> psroiPooling = std::make_shared<ngraph::op::v0::PSROIPooling>(paramOuts[0],
                                                                                                constNode,
                                                                                                outputDim,
                                                                                                groupSize_,
                                                                                                spatialScale_,
                                                                                                spatialBinsX_,
                                                                                                spatialBinsY_,
                                                                                                mode_);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(psroiPooling)};
    function = std::make_shared<ngraph::Function>(results, params, "psroi_pooling");
}
}  // namespace LayerTestsDefinitions
