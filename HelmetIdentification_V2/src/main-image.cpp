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

#include "utils.h"

#include <iostream>
#include <fstream>
#include <time.h>
using namespace std;

// 如果在200DK上运行就改为 USE_200DK
#define USE_DVPP

APP_ERROR readImage(std::string imgPath, MxBase::Image &image, MxBase::ImageProcessor &imageProcessor)
{
    APP_ERROR ret;
#ifdef USE_DVPP
    // if USE DVPP
    ret = imageProcessor.Decode(imgPath, image);
#endif
#ifdef USE_200DK
    std::shared_ptr<uint8_t> dataPtr;
    uint32_t dataSize;
    // Get image data to memory, this method can be substituted or designed by yourself!
    std::ifstream file;
    file.open(imgPath.c_str(), std::ios::binary);
    if (!file)
    {
        LogInfo << "Invalid file.";
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
    if (ret != APP_ERR_OK)
    {
        LogError << "Getimage failed, ret=" << ret;
        return ret;
    }
    ret = imageProcessor.Decode(dataPtr, dataSize, image, MxBase::ImageFormat::YUV_SP_420);
    // endif
#endif
    if (ret != APP_ERR_OK)
    {
        LogError << "Decode failed, ret=" << ret;
        return ret;
    }
}

void postProcess(std::vector<MxBase::Tensor> modelOutputs, std::shared_ptr<MxBase::Yolov3PostProcess> postProcessorDptr,
                 MxBase::Size originalSize, MxBase::Size resizeSize, std::string outFileName)
{
    MxBase::ResizedImageInfo imgInfo;
    auto shape = modelOutputs[0].GetShape();
    imgInfo.widthOriginal = originalSize.width;
    imgInfo.heightOriginal = originalSize.height;
    imgInfo.widthResize = resizeSize.width;
    imgInfo.heightResize = resizeSize.height;
    imgInfo.resizeType = MxBase::RESIZER_MS_KEEP_ASPECT_RATIO;
    float resizeRate = originalSize.width > originalSize.height ? (originalSize.width * 1.0 / videoInfo::YOLOV5_RESIZE) : (originalSize.height * 1.0 / videoInfo::YOLOV5_RESIZE);
    imgInfo.keepAspectRatioScaling = 1 / resizeRate;
    std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
    imageInfoVec.push_back(imgInfo);
    // make postProcess inputs
    std::vector<MxBase::TensorBase> tensors;
    for (size_t i = 0; i < modelOutputs.size(); i++)
    {
        MxBase::MemoryData memoryData(modelOutputs[i].GetData(), modelOutputs[i].GetByteSize());
        MxBase::TensorBase tensorBase(memoryData, true, modelOutputs[i].GetShape(), MxBase::TENSOR_DTYPE_INT32);
        tensors.push_back(tensorBase);
    }
    std::vector<std::vector<MxBase::ObjectInfo>> objectInfos;
    postProcessorDptr->Process(tensors, objectInfos, imageInfoVec);
    std::cout << "===---> Size of objectInfos is " << objectInfos.size() << std::endl;
    std::ofstream out(outFileName);
    for (size_t i = 0; i < objectInfos.size(); i++)
    {
        std::cout << "objectInfo-" << i << " , Size:" << objectInfos[i].size() << std::endl;
        for (size_t j = 0; j < objectInfos[i].size(); j++)
        {
            std::cout << std::endl
                      << "*****objectInfo-" << i << ":" << j << std::endl;
            std::cout << "x0 is " << objectInfos[i][j].x0 << std::endl;
            std::cout << "y0 is " << objectInfos[i][j].y0 << std::endl;
            std::cout << "x1 is " << objectInfos[i][j].x1 << std::endl;
            std::cout << "y1 is " << objectInfos[i][j].y1 << std::endl;
            std::cout << "confidence is " << objectInfos[i][j].confidence << std::endl;
            std::cout << "classId is " << objectInfos[i][j].classId << std::endl;
            std::cout << "className is " << objectInfos[i][j].className << std::endl;

            // 这里的out是输出流，不是std::cout
            out << objectInfos[i][j].className;
            out << " ";
            out << objectInfos[i][j].confidence;
            out << " ";
            out << objectInfos[i][j].x0;
            out << " ";
            out << objectInfos[i][j].y0;
            out << " ";
            out << objectInfos[i][j].x1;
            out << " ";
            out << objectInfos[i][j].y1;
            out << "\n";
        }
    }
    out.close();
}

APP_ERROR main(int argc, char *argv[])
{
    APP_ERROR ret;

    // global init
    ret = MxBase::MxInit();
    if (ret != APP_ERR_OK)
    {
        LogError << "MxInit failed, ret=" << ret << ".";
    }
    // 检测是否输入了文件路径
    if (argc <= 1)
    {
        LogWarn << "Please input image path, such as 'test.jpg'.";
        return APP_ERR_OK;
    }
    std::string num = argv[1];

    // imageProcess init
    MxBase::ImageProcessor imageProcessor(videoInfo::DEVICE_ID);
    // model init
    MxBase::Model yoloModel(videoInfo::modelPath, videoInfo::DEVICE_ID);
    // postprocessor init
    std::map<std::string, std::string> postConfig;
    postConfig.insert(std::pair<std::string, std::string>("postProcessConfigPath", videoInfo::configPath));
    postConfig.insert(std::pair<std::string, std::string>("labelPath", videoInfo::labelPath));
    std::shared_ptr<MxBase::Yolov3PostProcess> postProcessorDptr = std::make_shared<MxBase::Yolov3PostProcess>();
    postProcessorDptr->Init(postConfig);

    std::string imgPath = "dataSet/TestImages/" + num + ".jpg";
    // 读取图片
    MxBase::Image image;
    readImage(imgPath, image, imageProcessor);

    // 缩放图片
    MxBase::Size originalSize = image.GetOriginalSize();
    MxBase::Size resizeSize = MxBase::Size(videoInfo::YOLOV5_RESIZE, videoInfo::YOLOV5_RESIZE);
    MxBase::Image resizedImage = resizeKeepAspectRatioFit(originalSize.width, originalSize.height, resizeSize.width, resizeSize.height, image, imageProcessor);

    // 模型推理
    MxBase::Tensor tensorImg = resizedImage.ConvertToTensor();
    tensorImg.ToDevice(videoInfo::DEVICE_ID);
    std::vector<MxBase::Tensor> inputs;
    inputs.push_back(tensorImg);
    std::vector<MxBase::Tensor> modelOutputs = yoloModel.Infer(inputs);
    for (auto output : modelOutputs)
    {
        output.ToHost();
    }

    // 后处理
    std::string outFileName = "dataSet/V2txt/" + num + ".txt";
    postProcess(modelOutputs, postProcessorDptr, originalSize, resizeSize, outFileName);

    return APP_ERR_OK;
}/*
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

#include "utils.h"

#include <iostream>
#include <fstream>
#include <time.h>
using namespace std;

// 如果在200DK上运行就改为 USE_200DK
#define USE_DVPP

APP_ERROR readImage(std::string imgPath, MxBase::Image &image, MxBase::ImageProcessor &imageProcessor)
{
    APP_ERROR ret;
#ifdef USE_DVPP
    // if USE DVPP
    ret = imageProcessor.Decode(imgPath, image);
#endif
#ifdef USE_200DK
    std::shared_ptr<uint8_t> dataPtr;
    uint32_t dataSize;
    // Get image data to memory, this method can be substituted or designed by yourself!
    std::ifstream file;
    file.open(imgPath.c_str(), std::ios::binary);
    if (!file)
    {
        LogInfo << "Invalid file.";
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
    if (ret != APP_ERR_OK)
    {
        LogError << "Getimage failed, ret=" << ret;
        return ret;
    }
    ret = imageProcessor.Decode(dataPtr, dataSize, image, MxBase::ImageFormat::YUV_SP_420);
    // endif
#endif
    if (ret != APP_ERR_OK)
    {
        LogError << "Decode failed, ret=" << ret;
        return ret;
    }
}

void postProcess(std::vector<MxBase::Tensor> modelOutputs, std::shared_ptr<MxBase::Yolov3PostProcess> postProcessorDptr,
                 MxBase::Size originalSize, MxBase::Size resizeSize, std::string outFileName)
{
    MxBase::ResizedImageInfo imgInfo;
    auto shape = modelOutputs[0].GetShape();
    imgInfo.widthOriginal = originalSize.width;
    imgInfo.heightOriginal = originalSize.height;
    imgInfo.widthResize = resizeSize.width;
    imgInfo.heightResize = resizeSize.height;
    imgInfo.resizeType = MxBase::RESIZER_MS_KEEP_ASPECT_RATIO;
    float resizeRate = originalSize.width > originalSize.height ? (originalSize.width * 1.0 / videoInfo::YOLOV5_RESIZE) : (originalSize.height * 1.0 / videoInfo::YOLOV5_RESIZE);
    imgInfo.keepAspectRatioScaling = 1 / resizeRate;
    std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
    imageInfoVec.push_back(imgInfo);
    // make postProcess inputs
    std::vector<MxBase::TensorBase> tensors;
    for (size_t i = 0; i < modelOutputs.size(); i++)
    {
        MxBase::MemoryData memoryData(modelOutputs[i].GetData(), modelOutputs[i].GetByteSize());
        MxBase::TensorBase tensorBase(memoryData, true, modelOutputs[i].GetShape(), MxBase::TENSOR_DTYPE_INT32);
        tensors.push_back(tensorBase);
    }
    std::vector<std::vector<MxBase::ObjectInfo>> objectInfos;
    postProcessorDptr->Process(tensors, objectInfos, imageInfoVec);
    std::cout << "===---> Size of objectInfos is " << objectInfos.size() << std::endl;
    std::ofstream out(outFileName);
    for (size_t i = 0; i < objectInfos.size(); i++)
    {
        std::cout << "objectInfo-" << i << " , Size:" << objectInfos[i].size() << std::endl;
        for (size_t j = 0; j < objectInfos[i].size(); j++)
        {
            std::cout << std::endl
                      << "*****objectInfo-" << i << ":" << j << std::endl;
            std::cout << "x0 is " << objectInfos[i][j].x0 << std::endl;
            std::cout << "y0 is " << objectInfos[i][j].y0 << std::endl;
            std::cout << "x1 is " << objectInfos[i][j].x1 << std::endl;
            std::cout << "y1 is " << objectInfos[i][j].y1 << std::endl;
            std::cout << "confidence is " << objectInfos[i][j].confidence << std::endl;
            std::cout << "classId is " << objectInfos[i][j].classId << std::endl;
            std::cout << "className is " << objectInfos[i][j].className << std::endl;

            // 这里的out是输出流，不是std::cout
            out << objectInfos[i][j].className;
            out << " ";
            out << objectInfos[i][j].confidence;
            out << " ";
            out << objectInfos[i][j].x0;
            out << " ";
            out << objectInfos[i][j].y0;
            out << " ";
            out << objectInfos[i][j].x1;
            out << " ";
            out << objectInfos[i][j].y1;
            out << "\n";
        }
    }
    out.close();
}

APP_ERROR main(int argc, char *argv[])
{
    APP_ERROR ret;

    // global init
    ret = MxBase::MxInit();
    if (ret != APP_ERR_OK)
    {
        LogError << "MxInit failed, ret=" << ret << ".";
    }
    // 检测是否输入了文件路径
    if (argc <= 1)
    {
        LogWarn << "Please input image path, such as 'test.jpg'.";
        return APP_ERR_OK;
    }
    std::string num = argv[1];

    // imageProcess init
    MxBase::ImageProcessor imageProcessor(videoInfo::DEVICE_ID);
    // model init
    MxBase::Model yoloModel(videoInfo::modelPath, videoInfo::DEVICE_ID);
    // postprocessor init
    std::map<std::string, std::string> postConfig;
    postConfig.insert(std::pair<std::string, std::string>("postProcessConfigPath", videoInfo::configPath));
    postConfig.insert(std::pair<std::string, std::string>("labelPath", videoInfo::labelPath));
    std::shared_ptr<MxBase::Yolov3PostProcess> postProcessorDptr = std::make_shared<MxBase::Yolov3PostProcess>();
    postProcessorDptr->Init(postConfig);

    std::string imgPath = "dataSet/TestImages/" + num + ".jpg";
    // 读取图片
    MxBase::Image image;
    readImage(imgPath, image, imageProcessor);

    // 缩放图片
    MxBase::Size originalSize = image.GetOriginalSize();
    MxBase::Size resizeSize = MxBase::Size(videoInfo::YOLOV5_RESIZE, videoInfo::YOLOV5_RESIZE);
    MxBase::Image resizedImage = resizeKeepAspectRatioFit(originalSize.width, originalSize.height, resizeSize.width, resizeSize.height, image, imageProcessor);

    // 模型推理
    MxBase::Tensor tensorImg = resizedImage.ConvertToTensor();
    tensorImg.ToDevice(videoInfo::DEVICE_ID);
    std::vector<MxBase::Tensor> inputs;
    inputs.push_back(tensorImg);
    std::vector<MxBase::Tensor> modelOutputs = yoloModel.Infer(inputs);
    for (auto output : modelOutputs)
    {
        output.ToHost();
    }

    // 后处理
    std::string outFileName = "dataSet/V2txt/" + num + ".txt";
    postProcess(modelOutputs, postProcessorDptr, originalSize, resizeSize, outFileName);

    return APP_ERR_OK;
}