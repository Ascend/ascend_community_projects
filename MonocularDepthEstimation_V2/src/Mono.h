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

#ifndef DEPTHESTIMATION
#define DEPTHESTIMATION
#include "MxBase/MxBase.h"
#include "opencv2/opencv.hpp"

struct V2Param
{
    uint32_t deviceId;
    std::string modelPath;

    V2Param() {}
    V2Param(const uint32_t &deviceId, const std::string &modelPath)
        : deviceId(deviceId), modelPath(modelPath) {}
};

class DepthEstimation
{
public:
    explicit DepthEstimation(const V2Param &v2Param);
    APP_ERROR GetImage(const std::string &imgPath, std::shared_ptr<uint8_t> &dataPtr, uint32_t &dataSize);
    APP_ERROR ReadImage(const std::string &imgPath, MxBase::Image &decodedImage);
    APP_ERROR Resize(const MxBase::Image &inputImage, MxBase::Image &resizedImage);
    APP_ERROR Infer(MxBase::Image &inputImage, std::vector<MxBase::Tensor> &outputs);

    APP_ERROR PostProcess(cv::Mat &outputImage, const std::vector<MxBase::Tensor> &outputs, const std::string &mode = "run");

protected:
    uint32_t deviceId = 0;
    std::shared_ptr<MxBase::ImageProcessor> imageProcessorDptr;
    std::shared_ptr<MxBase::Model> modelDptr;

private:
};

#endif // DEPTHESTIMATION
