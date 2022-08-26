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

#ifndef MXBASE_REFINEDETDETECTION_H
#define MXBASE_REFINEDETDETECTION_H

#include <RefineDetPostProcess.h>
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"

struct InitParam {
    uint32_t deviceId;
    std::string labelPath;
    bool checkTensor;
    std::string modelPath;
    uint32_t classNum;
    uint32_t biasesNum;
    std::string biases;
    std::string objectnessThresh;
    std::string iouThresh;
    std::string scoreThresh;
    uint32_t yoloType;
    uint32_t modelType;
    uint32_t inputType;
    uint32_t anchorDim;
};

class RefineDetDetection {
public:
    APP_ERROR Init(const InitParam &initParam);
    APP_ERROR DeInit();
    APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs, std::vector<MxBase::TensorBase> &outputs);
    APP_ERROR PostProcess(const MxBase::TensorBase &tensor, const std::vector<MxBase::TensorBase> &outputs,
                          std::vector<std::vector<MxBase::ObjectInfo>> &objInfos);
    APP_ERROR Process(const std::string &imgPath);
protected:
    APP_ERROR ReadImage(const std::string &imgPath, MxBase::TensorBase &tensor);
    APP_ERROR Resize(const MxBase::TensorBase &inputTensor, MxBase::TensorBase &outputTensor);
    APP_ERROR LoadLabels(const std::string &labelPath, std::map<int, std::string> &labelMap);
    APP_ERROR WriteResult(MxBase::TensorBase &tensor,
                         const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos);
    void SetRefineDetPostProcessConfig(const InitParam &initParam, std::map<std::string, std::shared_ptr<void>> &config);
private:
    std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_; // 封装DVPP基本编码、解码、扣图功能
    std::shared_ptr<MxBase::ModelInferenceProcessor> model_; // 模型推理功能处理
    std::shared_ptr<RefineDetPostProcess> post_;
    MxBase::ModelDesc modelDesc_ = {}; // 模型描述信息
    std::map<int, std::string> labelMap_ = {};
    uint32_t deviceId_ = 0;
};
#endif