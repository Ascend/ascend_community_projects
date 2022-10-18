/*
 * Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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

#include "YoloxPostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include <algorithm>
namespace {
    auto g_uint8Deleter = [] (uint8_t *p) { };
}

namespace MxBase {
// 用config文件初始化YoloxPostProcess类内成员变量
    APP_ERROR YoloxPostProcess::Init(const std::map <std::string, std::shared_ptr<void>> &postConfig) {
        LogDebug << "Start to Init YoloxPostProcess.";
        APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
            return ret;
        }
        std::string str_strides;
        std::string str_inputSize;
        configData_.GetFileValue<std::string>("STRIDES", str_strides);
        configData_.GetFileValue<float>("OBJECTNESS_THRESH", objectnessThresh_);
        configData_.GetFileValue<int>("STRIDENUM", stridesNum_);
        configData_.GetFileValue<std::string>("INPUT_SIZE", str_inputSize);
        configData_.GetFileValue<int>("CLASS_NUM", num_class_);
        configData_.GetFileValue<float>("IOU_THRESH", iouThresh_);
        ret = GetStrides(str_strides);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Failed to get biases.";
            return ret;
        }
        ret = GetInput(str_inputSize);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "Failed to get input.";
            return ret;
        }
        LogDebug << "End to Init YoloxPostProcess.";
        return APP_ERR_OK;
    }

    APP_ERROR YoloxPostProcess::DeInit() {
        return APP_ERR_OK;
    }

    void YoloxPostProcess::GenerateGridsAndStride(std::vector<int> &strides,
                                                  std::vector <DefaultValues::GridAndStride> &grid_strides) {
        int target_size = inputSize_[0];
        
        for (int i = 0; i < (int) strides.size(); i++) {
            int stride = strides[i];
            int num_grid = target_size / stride;
            for (int g1 = 0; g1 < num_grid; g1++) {
                for (int g0 = 0; g0 < num_grid; g0++) {
                    DefaultValues::GridAndStride gs;
                    gs.grid0 = g0;
                    gs.grid1 = g1;
                    gs.stride = stride;
                    grid_strides.push_back(gs);
                }
            }
        }
    }

    void YoloxPostProcess::GenerateYoloxProposals(const std::vector <TensorBase> &tensors,
                                                  std::vector <DefaultValues::GridAndStride> grid_strides,
                                                  float prob_threshold,
                                                  std::vector <ObjectInfo> &objects,
                                                  ResizedImageInfo resizedImageInfos) {
        int widthResize = resizedImageInfos.widthResize;
        int heightResize = resizedImageInfos.heightResize;
        int widthOriginal = resizedImageInfos.widthOriginal;
        int heightOriginal = resizedImageInfos.heightOriginal;
        if (widthResize == widthOriginal && heightResize == heightOriginal) {
            widthResize = inputSize_[0];
            heightResize = inputSize_[1];
        }
        float widthscale = static_cast<float>(widthResize) / static_cast<float>(widthOriginal);
        float heightscale = static_cast<float>(heightResize) / static_cast<float>(heightOriginal);
        float scale = widthscale > heightscale ? heightscale : widthscale;

        auto shapeReg = tensors[0].GetShape();
        int num_anchors = shapeReg[1];

        auto DataPtr = (uint8_t *) tensors[0].GetBuffer();
        std::shared_ptr<void> DataPointer;
        DataPointer.reset(DataPtr, g_uint8Deleter);
        int idx = 0;
        for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
            const int grid0 = grid_strides[anchor_idx].grid0;
            const int grid1 = grid_strides[anchor_idx].grid1;
            const int stride = grid_strides[anchor_idx].stride;

            float x_center = (static_cast<float *>(DataPointer.get())[idx + 0] + grid0) * stride;
            float y_center = (static_cast<float *>(DataPointer.get())[idx + 1] + grid1) * stride;
            float w = exp(static_cast<float *>(DataPointer.get())[idx + 2]) * stride;
            float h = exp(static_cast<float *>(DataPointer.get())[idx + 3]) * stride;
            float x0 = x_center - w * 0.5f;
            float y0 = y_center - h * 0.5f;
            float x1 = x_center + w * 0.5f;
            float y1 = y_center + h * 0.5f;
            float box_objectness = static_cast<float *>(DataPointer.get())[idx + 4];
            for (int class_idx = 0; class_idx < num_class_; class_idx++) {
                float box_cls_score = static_cast<float *>(DataPointer.get())[idx + 5 + class_idx];
                float box_prob = box_objectness * box_cls_score;
                if (box_prob > prob_threshold) {
                    MxBase::ObjectInfo obj;
                    obj.x0 = x0 / scale;
                    obj.y0 = y0 / scale;
                    obj.x1 = x1 / scale;
                    obj.y1 = y1 / scale;
                    obj.classId = class_idx;
                    obj.className = configData_.GetClassName(class_idx);
                    obj.confidence = box_prob;

                    objects.push_back(obj);
                }
            } // class loop
            idx += DefaultValues::LOGIT_AND_SCORE + num_class_;
        }
    }

// 将处理好的推理结果放入ObjectInfo
    void YoloxPostProcess::ObjectDetectionOutput(const std::vector <TensorBase> &tensors,
                                                 std::vector <std::vector<ObjectInfo>> &objectInfos,
                                                 const std::vector <ResizedImageInfo> &resizedImageInfos) {
        LogDebug << "YoloxPostProcess start to write results.";
        if (tensors.size() == 0) {
            return;
        }
        auto shape = tensors[0].GetShape();
        if (shape.size() == 0) {
            return;
        }
        uint32_t batchSize = shape[0];
        std::vector<int> strides(strideArr_);
        for (uint32_t i = 0; i < batchSize; i++) {
            std::vector <DefaultValues::GridAndStride> grid_strides;
            GenerateGridsAndStride(strides, grid_strides);
            std::vector <ObjectInfo> objectInfo;
            GenerateYoloxProposals(tensors, grid_strides, objectnessThresh_, objectInfo, resizedImageInfos[i]);
            MxBase::NmsSort(objectInfo, iouThresh_);
            objectInfos.push_back(objectInfo);
        }
        LogDebug << "YoloxPostProcess write results successed.";
    }

    APP_ERROR YoloxPostProcess::Process(const std::vector <TensorBase> &tensors,
                                        std::vector <std::vector<ObjectInfo>> &objectInfos,
                                        const std::vector <ResizedImageInfo> &resizedImageInfos,
                                        const std::map <std::string, std::shared_ptr<void>> &paramMap) {
        LogDebug << "Start to Process YoloxPostProcess.";
        APP_ERROR ret = APP_ERR_OK;
        if (resizedImageInfos.size() == 0) {
            ret = APP_ERR_INPUT_NOT_MATCH;
            LogError << GetError(ret) << "resizedImageInfos is not provided which is necessary for YoloxPostProcess.";
            return ret;
        }
        auto inputs = tensors;
        ret = CheckAndMoveTensors(inputs);
        if (ret != APP_ERR_OK) {
            LogError << GetError(ret) << "CheckAndMoveTensors failed.";
            return ret;
        }

        ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);

        LogDebug << "End to Process YoloxPostProcess.";
        return APP_ERR_OK;
    }

    // 将strides字符串解析为int型数组存入strideArr_中
    APP_ERROR YoloxPostProcess::GetStrides(std::string &strStrides) {
        if (stridesNum_ <= 0) {
            LogError << GetError(APP_ERR_COMM_INVALID_PARAM) << "Failed to get stridesNum (" << stridesNum_ << ").";
            return APP_ERR_COMM_INVALID_PARAM;
        }
        strideArr_.clear();
        int i = 0;
        int num = strStrides.find(",");
        while (num >= 0 && i < stridesNum_) {
            std::string tmp = strStrides.substr(0, num);
            num++;
            strStrides = strStrides.substr(num, strStrides.size());
            strideArr_.push_back(stoi(tmp));
            i++;
            num = strStrides.find(",");
        }
        if (i != stridesNum_ - 1 || strStrides.size() <= 0) {
            LogError << GetError(APP_ERR_COMM_INVALID_PARAM) << "stridesNum (" << stridesNum_
                     << ") is not equal to total number of biases (" << strStrides << ").";
            return APP_ERR_COMM_INVALID_PARAM;
        }
        strideArr_.push_back(stof(strStrides));
        return APP_ERR_OK;
    }
    // 将inputSize字符串解析为int型数组存入inputSize_中
    APP_ERROR YoloxPostProcess::GetInput(std::string &strInput) {
        inputSize_.clear();
        int num = strInput.find(",");
        std::string height = strInput.substr(0, num);
        std::string width = strInput.substr(num + 1);
        inputSize_.push_back(std::stoi(height));
        inputSize_.push_back(std::stoi(width));
        return APP_ERR_OK;
    }

    extern "C" {
    std::shared_ptr <MxBase::YoloxPostProcess> GetObjectInstance() {
        LogInfo << "Begin to get YoloxPostProcess instance.";
        auto instance = std::make_shared<MxBase::YoloxPostProcess>();
        LogInfo << "End to get YoloxPostProcess instance.";
        return instance;
        }
    }
}