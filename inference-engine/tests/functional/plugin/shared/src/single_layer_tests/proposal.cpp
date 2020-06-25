// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "single_layer_tests/proposal.hpp"

namespace LayerTestsDefinitions {

const normalize_type normalize = true;
const feat_stride_type feat_stride = 1;
const box_size_scale_type box_size_scale = 2.0f;
const box_coordinate_scale_type box_coordinate_scale = 2.0f;

std::string ProposalLayerTest::getTestCaseName(testing::TestParamInfo<proposalLayerTestParamsSet> obj) {
    proposalSpecificParams proposalParams;

    std::string targetDevice;
    std::tie(proposalParams, targetDevice) = obj.param;

    base_size_type base_size;
    pre_nms_topn_type pre_nms_topn;
    post_nms_topn_type post_nms_topn;
    nms_thresh_type nms_thresh;
    min_size_type min_size;
    ratio_type ratio;
    scale_type scale;
    clip_before_nms_type clip_before_nms;
    clip_after_nms_type clip_after_nms;
    framework_type framework;
    std::tie(base_size, pre_nms_topn,
             post_nms_topn,
             nms_thresh,
             min_size,
             ratio,
             scale,
             clip_before_nms,
             clip_after_nms,
             framework) = proposalParams;

    std::ostringstream result;
    result << "base_size=" << base_size << "_";
    result << "pre_nms_topn=" << pre_nms_topn << "_";
    result << "post_nms_topn=" << post_nms_topn << "_";
    result << "nms_thresh=" << nms_thresh << "_";
    result << "feat_stride=" << feat_stride << "_";
    result << "min_size=" << min_size << "_";
    result << "ratio = " << CommonTestUtils::vec2str(ratio) << "_";
    result << "scale = " << CommonTestUtils::vec2str(scale) << "_";
    result << "clip_before_nms=" << clip_before_nms << "_";
    result << "clip_after_nms=" << clip_after_nms << "_";
    result << "normalize=" << normalize << "_";
    result << "box_size_scale=" << box_size_scale << "_";
    result << "box_coordinate_scale=" << box_coordinate_scale << "_";
    result << "framework=" << framework << "_";
    result << "targetDevice=" << targetDevice;
    auto string = result.str();
    std::replace(string.begin(), string.end(), '-', '_');
    std::replace(string.begin(), string.end(), '.', '_');
    return string;
}

void ProposalLayerTest::SetUp() {
    proposalSpecificParams proposalParams;

    std::tie(proposalParams, targetDevice) = this->GetParam();
    base_size_type base_size;
    pre_nms_topn_type pre_nms_topn;
    post_nms_topn_type post_nms_topn;
    nms_thresh_type nms_thresh;
    min_size_type min_size;
    ratio_type ratio;
    scale_type scale;
    clip_before_nms_type clip_before_nms;
    clip_after_nms_type clip_after_nms;
    framework_type framework;

    std::tie(base_size, pre_nms_topn,
             post_nms_topn,
             nms_thresh,
             min_size,
             ratio,
             scale,
             clip_before_nms,
             clip_after_nms,
             framework) = proposalParams;

    size_t bottom_w = base_size;
    size_t bottom_h = base_size;
    size_t num_anchors = ratio.size() * scale.size();

    std::vector<size_t> scoresShape = {1, 2 * num_anchors, bottom_h, bottom_w};
    std::vector<size_t> boxesShape  = {1, 4 * num_anchors, bottom_h, bottom_w};
    std::vector<size_t> imageInfoShape = {3};

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(InferenceEngine::Precision::FP16);
    auto params = ngraph::builder::makeParams(ngPrc, {{"scores", scoresShape}, {"boxes", boxesShape}, {"image_info", imageInfoShape}});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    auto proposal = std::dynamic_pointer_cast<ngraph::opset1::Proposal>(
             ngraph::builder::makeProposal(paramOuts[0], paramOuts[1], paramOuts[2], ngPrc,
                                           base_size,
                                           pre_nms_topn,
                                           post_nms_topn,
                                           nms_thresh,
                                           feat_stride,
                                           min_size,
                                           ratio,
                                           scale,
                                           clip_before_nms,
                                           clip_after_nms,
                                           normalize,
                                           box_size_scale,
                                           box_coordinate_scale,
                                           framework));

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(proposal)};
    function = std::make_shared<ngraph::Function>(results, params, "proposal");
}

InferenceEngine::Blob::Ptr ProposalLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    InferenceEngine::Blob::Ptr blobPtr;

    const std::string name = info.name();
    if (name == "scores") {
        blobPtr = FuncTestUtils::createAndFillBlobFloat(info.getTensorDesc(), 1, 0, 1000, 8234231);
    } else if (name == "boxes") {
        blobPtr = FuncTestUtils::createAndFillBlobFloatNormalDistribution(info.getTensorDesc(), 0.0f, 0.2f, 7235346);
    } else if (name == "image_info") {
        const float image_info[] = {225.0f, 225.0f, 1.0f};
        blobPtr = FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), image_info, 3);
    }

    return blobPtr;
}

// TODO: for validation, reference version is required (#28373)
void ProposalLayerTest::Validate() {}

TEST_P(ProposalLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
