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

#ifndef YOLOX_YOLOXPOSTPROCESS_H
#define YOLOX_YOLOXPOSTPROCESS_H
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
namespace DefaultValues {
    // 回归框坐标和预测得分
    const float LOGIT_AND_SCORE = 5;
    const float DEFAULT_OBJECTNESS_THRESH = 0.3;
    const float DEFAULT_IOU_THRESH = 0.45;
    const int NUM_CLASSES = 80;
    const int STRIDESNUM = 3;
    // 保存每个像素点的坐标与步长信息
    struct GridAndStride
    {
        int grid0;
        int grid1;
        int stride;
    };
}
// YOLOX继承ObjectPostProcessBase
namespace MxBase {
    class YoloxPostProcess : public MxBase::ObjectPostProcessBase {
    public:
        YoloxPostProcess() = default;

        ~YoloxPostProcess() = default;

        YoloxPostProcess(const YoloxPostProcess &other) = default;

        YoloxPostProcess &operator=(const YoloxPostProcess &other) = default;

        APP_ERROR Init(const std::map <std::string, std::shared_ptr<void>> &postConfig) override;

        APP_ERROR DeInit() override;

        APP_ERROR Process(const std::vector <MxBase::TensorBase> &tensors,
                          std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                          const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {},
                          const std::map <std::string, std::shared_ptr<void>> &paramMap = {}) override;

    protected:
        bool IsValidTensors(const std::vector <MxBase::TensorBase> &tensors) const;

        void ObjectDetectionOutput(const std::vector <MxBase::TensorBase> &tensors,
                                   std::vector <std::vector<MxBase::ObjectInfo>> &objectInfos,
                                   const std::vector <MxBase::ResizedImageInfo> &resizedImageInfos = {});

        void GenerateGridsAndStride(std::vector<int> &strides,
                                       std::vector <DefaultValues::GridAndStride> &grid_strides);

        void GenerateYoloxProposals(const std::vector <TensorBase> &tensors,
                                      std::vector <DefaultValues::GridAndStride> grid_strides,
                                      float prob_threshold,
                                      std::vector <ObjectInfo> &objects,
                                      ResizedImageInfo resizedImageInfos);

        APP_ERROR GetStrides(std::string &strStrides);
        APP_ERROR GetInput(std::string &strInput);

    protected:
        float objectnessThresh_ = DefaultValues::DEFAULT_OBJECTNESS_THRESH; // Threshold of objectness value
        float iouThresh_ = DefaultValues::DEFAULT_IOU_THRESH; // Non-Maximum Suppression threshold
        int num_class_ = DefaultValues::NUM_CLASSES;
        int stridesNum_ = DefaultValues::STRIDESNUM;
        std::vector<int> strideArr_ = {};
        std::vector<int> inputSize_ = {};
    };

    extern "C" {
    std::shared_ptr <MxBase::YoloxPostProcess> GetObjectInstance();
    }
}

#endif // YOLOX_YOLOXPOSTPROCESS_H
