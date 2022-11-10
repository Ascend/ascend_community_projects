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

#ifndef DETECTANDALIGN
#define DETECTANDALIGN
#include "ParamDataType.h"
#include "MxBase/MxBase.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include "MxBase/postprocess/include/ObjectPostProcessors/Yolov3PostProcess.h"
#include "opencv2/opencv.hpp"

class Yolo
{
public:
    explicit Yolo(const V2Param &v2Param);
    APP_ERROR Infer(MxBase::Image &inputImage, std::vector<MxBase::Tensor> &outputs);
    APP_ERROR PostProcess(std::vector<MxBase::Tensor> &outputs, std::vector<MxBase::Rect> &cropConfigVec);

protected:
    std::shared_ptr<MxBase::Model> modelDptr;
    std::shared_ptr<MxBase::Yolov3PostProcess> postProcessorDptr;

private:
    uint32_t deviceId = 0;
};

class FaceLandMark
{
public:
    explicit FaceLandMark(const V2Param &v2Param);
    APP_ERROR Infer(MxBase::Image &inputImage, std::vector<MxBase::Tensor> &outputs);
    APP_ERROR PostProcess(std::vector<MxBase::Tensor> &outputs, const cv::Mat &inputMat, cv::Mat &affinedMat);

protected:
    std::shared_ptr<MxBase::Model> modelDptr;

private:
    uint32_t deviceId = 0;
};

#endif
