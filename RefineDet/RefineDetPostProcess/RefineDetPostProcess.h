/*
 * Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef REFINEDET_POST_PROCESS_H
#define REFINEDET_POST_PROCESS_H
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include "opencv2/opencv.hpp"

const float DEFAULT_OBJECTNESS_THRESH = 0.01;
const float DEFAULT_CONFIDENCE_THRESH = 0.1;
const float DEFAULT_IOU_THRESH = 0.45;
const int DEFAULT_ANCHOR_DIM = 3;
const int DEFAULT_BIASES_NUM = 18;
const int DEFAULT_YOLO_TYPE = 3;
const int DEFAULT_YOLO_VERSION = 3;
const int RefineDet_VERSION = 3;
const int YOLOV4_VERSION = 4;
const int YOLOV5_VERSION = 5;

struct OutputLayer {
    size_t width;
    size_t height;
    float anchors[6];
};

struct NetInfo {
    int anchorDim;
    int classNum;
    int bboxDim;
    int netWidth;
    int netHeight;
};


class RefineDetPostProcess : public MxBase::ObjectPostProcessBase {
public:
    RefineDetPostProcess() = default;

    ~RefineDetPostProcess() = default;

    RefineDetPostProcess(const RefineDetPostProcess &other);

    RefineDetPostProcess &operator=(const RefineDetPostProcess &other);

    APP_ERROR Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) override;

    APP_ERROR DeInit() override;

    APP_ERROR Process(const std::vector<MxBase::TensorBase> &tensors, std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
                      const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos = {},
                      const std::map<std::string, std::shared_ptr<void>> &paramMap = {}) override;
protected:
    bool IsValidTensors(const std::vector<MxBase::TensorBase> &tensors) const;

    void ObjectDetectionOutput(const std::vector<MxBase::TensorBase> &tensors,
                               std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
                               const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos = {});

    void CompareProb(int& classID, float& maxProb, float classProb, int classNum);
    void CodeDuplicationSetDet(float& x, float& y, float& width, float& height);
    void SelectClassNCHW(std::shared_ptr<void> netout, NetInfo info, std::vector<MxBase::ObjectInfo>& detBoxes,
                         int stride, OutputLayer layer);
    void SelectClassNHWC(std::shared_ptr<void> netout, NetInfo info, std::vector<MxBase::ObjectInfo>& detBoxes,
                         int stride, OutputLayer layer);
    void SelectClassNCHWC(std::shared_ptr<void> netout, NetInfo info, std::vector<MxBase::ObjectInfo>& detBoxes,
                         int stride, OutputLayer layer);

    void GenerateBbox(std::vector<std::shared_ptr<void>> featLayerData,
                      std::vector<MxBase::ObjectInfo> &detBoxes, const std::vector<std::vector<size_t>>& featLayerShapes,
                      const int netWidth, const int netHeight);
    void GeneratePriorBox(cv::Mat &anchors);
    cv::Mat center_size(cv::Mat boxes);
    cv::Mat decode_for_loc(cv::Mat &loc, cv::Mat &prior);
    APP_ERROR GetBiases(std::string& strBiases);
    
    void generate_objectInfos(const std::vector <MxBase::TensorBase>& tensors,
                          std::vector <std::vector<MxBase::ObjectInfo>>& objectInfos,
                          const std::vector <MxBase::ResizedImageInfo>& resizedImageInfos,
                          cv::Mat& res);

protected:
    float objectnessThresh_ = DEFAULT_OBJECTNESS_THRESH; // Threshold of objectness value
    float confidenceThresh_ = DEFAULT_CONFIDENCE_THRESH;
    float iouThresh_ = DEFAULT_IOU_THRESH; // Non-Maximum Suppression threshold
    int anchorDim_ = DEFAULT_ANCHOR_DIM;
    int biasesNum_ = DEFAULT_BIASES_NUM; // RefineDet anchors, generate from train data, coco dataset
    int yoloType_ = DEFAULT_YOLO_TYPE;
    int modelType_ = 0;
    int yoloVersion_ = DEFAULT_YOLO_VERSION;
    int inputType_ = 0;
    std::vector<float> biases_ = {};
};
#endif