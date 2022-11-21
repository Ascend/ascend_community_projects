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
#include "Mono.h"
#include "MxBase/Log/Log.h"

// This option should be opened when using 200DK
#define USE_200DK

using namespace MxBase;

namespace
{
    const Size INPUT_SIZE(640, 480);
    const cv::Size OUTPUT_SIZE(640, 480);
}

DepthEstimation::DepthEstimation(const V2Param &v2Param)
{
    deviceId = v2Param.deviceId;
    std::string modelPath = v2Param.modelPath;
    APP_ERROR ret;

    // global init
    ret = MxInit();
    if (ret != APP_ERR_OK)
    {
        LogError << "MxInit failed, ret=" << ret << ".";
    }

    // imageProcess init
    imageProcessorDptr = std::make_shared<MxBase::ImageProcessor>(deviceId);
    if (imageProcessorDptr == nullptr)
    {
        LogError << "imageProcessorDptr nullptr";
    }

    // model init
    modelDptr = std::make_shared<MxBase::Model>(modelPath, deviceId);
    if (modelDptr == nullptr)
    {
        LogError << "modelDptr nullptr";
    }
};

APP_ERROR DepthEstimation::GetImage(const std::string &imgPath, std::shared_ptr<uint8_t> &dataPtr, uint32_t &dataSize)
{
    // Get image data to memory
    std::ifstream file;
    file.open(imgPath.c_str(), std::ios::binary);
    if (!file)
    {
        LogError << "Invalid file : " << imgPath;
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
}

APP_ERROR DepthEstimation::ReadImage(const std::string &imgPath, Image &decodedImage)
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

APP_ERROR DepthEstimation::Resize(const Image &decodedImage, Image &resizedImage)
{
    APP_ERROR ret;

    ret = imageProcessorDptr->Resize(decodedImage, INPUT_SIZE, resizedImage, Interpolation::HUAWEI_HIGH_ORDER_FILTER);
    if (ret != APP_ERR_OK)
    {
        LogError << "Resize failed, ret=" << ret;
        return ret;
    }

    return APP_ERR_OK;
};

APP_ERROR DepthEstimation::Infer(Image &inputImage, std::vector<Tensor> &outputs)
{
    APP_ERROR ret;

    // !move image to device!
    Tensor tensorImg = inputImage.ConvertToTensor();
    ret = tensorImg.ToDevice(deviceId);
    if (ret != APP_ERR_OK)
    {
        LogError << "ToDevice failed, ret=" << ret;
        return ret;
    }

    // make infer input
    std::vector<Tensor> inputs = {tensorImg};

    // do infer
    outputs = modelDptr->Infer(inputs);

    // !move result to host!
    for (size_t i = 0; i < outputs.size(); i++)
    {
        outputs[i].ToHost();
    }

    return APP_ERR_OK;
};

APP_ERROR DepthEstimation::PostProcess(cv::Mat &outputImage, const std::vector<MxBase::Tensor> &outputs, const std::string &mode)
{
    auto shape = outputs[1].GetShape();
    auto h = shape[2];
    auto w = shape[3];

    // convert tensor to OpenCV's mat
    outputImage = cv::Mat(cv::Size(w, h), CV_32FC1, (float32_t *)outputs[1].GetData());
    if (mode == "run")
    {
        cv::resize(outputImage, outputImage, OUTPUT_SIZE, 0, 0, cv::INTER_LINEAR);

        // normalize
        double value_min, value_max;
        cv::minMaxLoc(outputImage, &value_min, &value_max);
        if (value_min != value_max)
        {
            outputImage = outputImage - value_min;
            outputImage = outputImage / (value_max - value_min);
        }
        else
        {
            outputImage = outputImage * 0.0;
        }

        // convert to 0~255
        outputImage.convertTo(outputImage, CV_8UC3, 255, 0);
    }

    return APP_ERR_OK;
}
