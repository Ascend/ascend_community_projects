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

#ifndef ATTR_RECOGNITION
#define ATTR_RECOGNITION
#include "ParamDataType.h"
#include "MxBase/MxBase.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"

class AttrRecognition
{
public:
    explicit AttrRecognition(const V2Param &v2Param);
    APP_ERROR Infer(MxBase::Image &resizeImage, std::vector<MxBase::Tensor> &outputs);
    APP_ERROR PostProcess(std::vector<MxBase::Tensor> &outputs, std::vector<std::vector<MxBase::ClassInfo>> &classInfos);

protected:
    std::shared_ptr<MxBase::ImageProcessor> imageProcessorDptr;
    std::shared_ptr<MxBase::Model> modelDptr;

private:
    uint32_t deviceId = 0;
    bool softmaxFlag = false;
    uint32_t classNum = 40;
    uint32_t topK = 40;
    std::vector<std::string> labels;
};

#endif