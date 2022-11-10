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

#ifndef IMAGEPROCESS
#define IMAGEPROCESS
#include "MxBase/MxBase.h"
#include "opencv2/opencv.hpp"

class ImageProcess
{
public:
    std::shared_ptr<MxBase::ImageProcessor> imageProcessorDptr;
    explicit ImageProcess(const uint32_t &deviceID);
    APP_ERROR ReadImage(const std::string &imgPath, MxBase::Image &decodedImage);
    APP_ERROR ConvertMatToImage(const cv::Mat &inputMat, MxBase::Image &outputImage);
    APP_ERROR Resize(const MxBase::Image &inputImage, MxBase::Image &resizedImage,
                     const MxBase::Size &resizeConfig);
    APP_ERROR Crop(const MxBase::Image &inputImage, MxBase::Image &cropedImage,
                   const MxBase::Rect &cropRect, const double &expandRatio = 0.0);
    APP_ERROR Crop(const cv::Mat &inputMat, cv::Mat &cropedMat,
                   const MxBase::Rect &cropRect, double expandRatio = 0.0);

protected:
    APP_ERROR GetImage(const std::string &imgPath, std::shared_ptr<uint8_t> &dataPtr, uint32_t &dataSize);

private:
    uint32_t deviceId = 0;
};

#endif