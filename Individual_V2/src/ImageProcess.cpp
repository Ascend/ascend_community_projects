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

#include "fstream"
#include "ImageProcess.h"
#include "opencv2/opencv.hpp"

using namespace MxBase;

#define USE_200DK

ImageProcess::ImageProcess(const uint32_t &deviceID)
{
    deviceId = deviceID;

    // imageProcess init
    imageProcessorDptr = std::make_shared<MxBase::ImageProcessor>(deviceId);
    if (imageProcessorDptr == nullptr)
    {
        LogError << "imageProcessorDptr nullptr";
    }
}

APP_ERROR ImageProcess::GetImage(const std::string &imgPath, std::shared_ptr<uint8_t> &dataPtr, uint32_t &dataSize)
{
    // Get image data to memory
    std::ifstream file;
    file.open(imgPath.c_str(), std::ios::binary);
    if (!file)
    {
        LogError << "Invalid file.";
        return APP_ERR_COMM_INVALID_PARAM;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    char *p = (char *)malloc(content.size());
    memcpy(p, content.data(), content.size());
    auto deleter = [](void *p) -> void
    {
        free(p);
        p = nullptr;
    };

    dataPtr.reset(static_cast<uint8_t *>((void *)(p)), deleter);
    dataSize = content.size();

    file.close();
    return APP_ERR_OK;
};

APP_ERROR ImageProcess::ReadImage(const std::string &imgPath, MxBase::Image &decodedImage)
{
    APP_ERROR ret;

#ifdef USE_200DK
    std::shared_ptr<uint8_t> dataPtr;
    uint32_t dataSize;
    ret = GetImage(imgPath, dataPtr, dataSize);
    if (ret != APP_ERR_OK)
    {
        LogError << "Get image failed, ret=" << ret;
        return ret;
    }
    ret = imageProcessorDptr->Decode(dataPtr, dataSize, decodedImage, ImageFormat::YUV_SP_420);
#else
    ret = imageProcessorDptr->Decode(imgPath, decodedImage, ImageFormat::YUV_SP_420);
#endif
    if (ret != APP_ERR_OK)
    {
        LogError << "Decode failed, ret=" << ret;
        return ret;
    }

    return APP_ERR_OK;
};

APP_ERROR ImageProcess::ConvertMatToImage(const cv::Mat &inputMat, MxBase::Image &outputImage)
{
    APP_ERROR ret;

    std::vector<uint8_t> buffer;
    cv::imencode(".jpg", inputMat, buffer);
    std::string content(reinterpret_cast<char *>(&buffer[0]), buffer.size());

    char *p = (char *)malloc(content.size());
    memcpy(p, content.data(), content.size());
    auto deleter = [](void *p) -> void
    {
        free(p);
        p = nullptr;
    };

    std::shared_ptr<uint8_t> dataPtr;
    dataPtr.reset(static_cast<uint8_t *>((void *)(p)), deleter);
    uint32_t dataSize = content.size();

    ret = imageProcessorDptr->Decode(dataPtr, dataSize, outputImage);
    if (ret != APP_ERR_OK)
    {
        LogError << "Get image failed, ret=" << ret;
        return ret;
    }

    return APP_ERR_OK;
};

APP_ERROR ImageProcess::Resize(const MxBase::Image &inputImage, MxBase::Image &resizedImage,
                               const MxBase::Size &resizeConfig)
{
    APP_ERROR ret;

    ret = imageProcessorDptr->Resize(inputImage, resizeConfig, resizedImage);
    if (ret != APP_ERR_OK)
    {
        LogError << "Resize failed, ret=" << ret;
        return ret;
    }

    return APP_ERR_OK;
};

APP_ERROR ImageProcess::Crop(const MxBase::Image &inputImage, MxBase::Image &cropedImage,
                             const MxBase::Rect &cropRect, const double &expandRatio)
{
    APP_ERROR ret;

    // check whetrher cropRect is a valid rect
    if (cropRect.y1 <= cropRect.y0 || cropRect.x1 <= cropRect.x0)
    {
        LogError << "Object Detection Error.";
        return APP_ERR_INVALID_PARAM;
    }

    // expand crop rect
    auto x0 = cropRect.x0 - (cropRect.x1 - cropRect.x0) * expandRatio;
    auto y0 = cropRect.y0 - (cropRect.y1 - cropRect.y0) * expandRatio;
    auto x1 = cropRect.x1 + (cropRect.x1 - cropRect.x0) * expandRatio;
    auto y1 = cropRect.y1 + (cropRect.x1 - cropRect.x0) * expandRatio;

    // limit the crop rect for not crossing the border of the original image
    auto originalSize = inputImage.GetOriginalSize();
    Rect expandCropRect;
    expandCropRect.x0 = x0 >= 0 ? x0 : 0;
    expandCropRect.y0 = y0 >= 0 ? y0 : 0;
    expandCropRect.x1 = x1 <= originalSize.width ? x1 : originalSize.width;
    expandCropRect.y1 = y1 <= originalSize.height ? y1 : originalSize.height;

    ret = imageProcessorDptr->Crop(inputImage, expandCropRect, cropedImage);
    if (ret != APP_ERR_OK)
    {
        LogError << "Crop failed, ret=" << ret;
        return ret;
    }

    return APP_ERR_OK;
}

// crop func for OpenCV's mat
APP_ERROR ImageProcess::Crop(const cv::Mat &inputMat, cv::Mat &cropedMat,
                             const MxBase::Rect &cropRect, double expandRatio)
{
    // check whetrher cropRect is a valid rect
    if (cropRect.y1 <= cropRect.y0 || cropRect.x1 <= cropRect.x0)
    {
        LogError << "Invalid crop config,x1 <= x0 or y1<= y0.Could be caused by obejct detection error.";
        return APP_ERR_INVALID_PARAM;
    }

    // expand crop rect
    auto x0 = cropRect.x0 - (cropRect.x1 - cropRect.x0) * expandRatio;
    auto y0 = cropRect.y0 - (cropRect.y1 - cropRect.y0) * expandRatio;
    auto x1 = cropRect.x1 + (cropRect.x1 - cropRect.x0) * expandRatio;
    auto y1 = cropRect.y1 + (cropRect.x1 - cropRect.x0) * expandRatio;

    // limit the crop rect for not crossing the border of the original image
    auto originalSize = inputMat.size();
    Rect expandCropRect;
    expandCropRect.x0 = x0 >= 0 ? x0 : 0;
    expandCropRect.y0 = y0 >= 0 ? y0 : 0;
    expandCropRect.x1 = x1 <= originalSize.width ? x1 : originalSize.width;
    expandCropRect.y1 = y1 <= originalSize.height ? y1 : originalSize.height;

    cropedMat = inputMat(cv::Range(expandCropRect.y0, expandCropRect.y1), cv::Range(expandCropRect.x0, expandCropRect.x1));

    return APP_ERR_OK;
}