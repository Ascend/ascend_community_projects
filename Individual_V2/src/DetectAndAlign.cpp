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

#include "DetectAndAlign.h"

namespace
{
    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t VPC_H_ALIGN = 2;
    const uint32_t YOLOV3_RESIZE = 416;

    // Target point for affine transform,especially for size 112*112
    const std::vector<cv::Point2f> TARGET_POINTS = {
        cv::Point2f((30.2946 + 8.0000), 51.6963),
        cv::Point2f((65.5318 + 8.0000), 51.6963),
        cv::Point2f((48.0252 + 8.0000), 71.7366),
        cv::Point2f((33.5493 + 8.0000), 92.3655),
        cv::Point2f((62.7299 + 8.0000), 92.3655),
    };

    const uint32_t AFFINE_SIZE = 112;         // Target size for affine transform
    const uint32_t LANDMARK_NUMS = 5;         // Nums of landmark points.2 on eyes,1 on nose and the other on mouth.
    const uint32_t LANDMARK_OUTPUT_SIZE = 48; // Output size of landmark
};

Yolo::Yolo(const V2Param &v2Param)
{
    deviceId = v2Param.deviceId;
    std::string modelPath = v2Param.modelPath;
    std::string labelPath = v2Param.labelPath;
    std::string configPath = v2Param.configPath;

    // model init
    modelDptr = std::make_shared<MxBase::Model>(modelPath, deviceId);
    if (modelDptr == nullptr)
    {
        LogError << "modelDptr nullptr";
    }

    // postprocessor init
    std::map<std::string, std::string> postConfig;
    postConfig.insert(std::pair<std::string, std::string>("postProcessConfigPath", configPath));
    postConfig.insert(std::pair<std::string, std::string>("labelPath", labelPath));
    postProcessorDptr = std::make_shared<MxBase::Yolov3PostProcess>();
    if (postProcessorDptr == nullptr)
    {
        LogError << "postProcessorDptr nullptr";
    }
    postProcessorDptr->Init(postConfig);
};

APP_ERROR Yolo::Infer(MxBase::Image &inputImage, std::vector<MxBase::Tensor> &outputs)
{
    APP_ERROR ret;
    // !move image to device!
    MxBase::Tensor tensorImg = inputImage.ConvertToTensor();
    ret = tensorImg.ToDevice(deviceId);
    if (ret != APP_ERR_OK)
    {
        LogError << "ToDevice failed, ret=" << ret;
        return ret;
    }

    // make infer input
    std::vector<MxBase::Tensor> inputs = {tensorImg};
    // do infer
    outputs = modelDptr->Infer(inputs);

    // !move result to host!
    for (size_t i = 0; i < outputs.size(); i++)
    {
        outputs[i].ToHost();
    }

    return APP_ERR_OK;
};

APP_ERROR Yolo::PostProcess(std::vector<MxBase::Tensor> &outputs, std::vector<MxBase::Rect> &cropConfigVec)
{
    std::vector<MxBase::TensorBase> tensors;
    for (size_t i = 0; i < outputs.size(); i++)
    {
        MxBase::MemoryData memoryData(outputs[i].GetData(), outputs[i].GetByteSize());
        MxBase::TensorBase tensorBase(memoryData, true, outputs[i].GetShape(), MxBase::TENSOR_DTYPE_INT32);
        tensors.push_back(tensorBase);
    }
    std::vector<std::vector<MxBase::ObjectInfo>> objectInfos;

    auto shape = outputs[0].GetShape();
    MxBase::ResizedImageInfo imgInfo;
    imgInfo.widthOriginal = YOLOV3_RESIZE;
    imgInfo.heightOriginal = YOLOV3_RESIZE;
    imgInfo.widthResize = YOLOV3_RESIZE;
    imgInfo.heightResize = YOLOV3_RESIZE;
    imgInfo.resizeType = MxBase::RESIZER_STRETCHING;
    std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
    imageInfoVec.push_back(imgInfo);

    // do postProcess
    postProcessorDptr->Process(tensors, objectInfos, imageInfoVec);

    std::vector<uint32_t> faceId;
    // print result
    std::cout << "Size of objectInfos is " << objectInfos.size() << std::endl;
    for (size_t i = 0; i < objectInfos.size(); i++)
    {
        std::cout << "objectInfo-" << i << " ,Size:" << objectInfos[i].size() << std::endl;
        for (size_t j = 0; j < objectInfos[i].size(); j++)
        {
            if (objectInfos[i][j].className == "face")
            {
                faceId.push_back(j);

                std::cout << std::endl
                          << "*****objectInfo-" << i << ":" << j << std::endl;
                std::cout << "x0 is " << objectInfos[i][j].x0 << std::endl;
                std::cout << "y0 is " << objectInfos[i][j].y0 << std::endl;
                std::cout << "x1 is " << objectInfos[i][j].x1 << std::endl;
                std::cout << "y1 is " << objectInfos[i][j].y1 << std::endl;
                std::cout << "confidence is " << objectInfos[i][j].confidence << std::endl;
                std::cout << "classId is " << objectInfos[i][j].classId << std::endl;
                std::cout << "className is " << objectInfos[i][j].className << std::endl;
            }
        }
    }

    // get crop rect only for face
    cropConfigVec.resize(faceId.size());
    for (size_t i = 0; i < faceId.size(); i++)
    {
        cropConfigVec[i].x0 = objectInfos[0][faceId[i]].x0;
        cropConfigVec[i].y0 = objectInfos[0][faceId[i]].y0;
        cropConfigVec[i].x1 = objectInfos[0][faceId[i]].x1;
        cropConfigVec[i].y1 = objectInfos[0][faceId[i]].y1;
    }

    return APP_ERR_OK;
};

FaceLandMark::FaceLandMark(const V2Param &v2Param)
{
    deviceId = v2Param.deviceId;
    std::string modelPath = v2Param.modelPath;

    // model init
    modelDptr = std::make_shared<MxBase::Model>(modelPath, deviceId);
    if (modelDptr == nullptr)
    {
        LogError << "modelDptr nullptr";
    }
};

APP_ERROR FaceLandMark::Infer(MxBase::Image &inputImage,
                              std::vector<MxBase::Tensor> &outputs)
{
    APP_ERROR ret;
    // !move image to device!
    MxBase::Tensor tensorImg = inputImage.ConvertToTensor();
    ret = tensorImg.ToDevice(deviceId);
    if (ret != APP_ERR_OK)
    {
        LogError << "ToDevice failed, ret=" << ret;
        return ret;
    }

    // make infer input
    std::vector<MxBase::Tensor> inputs = {tensorImg};
    // do infer
    outputs = modelDptr->Infer(inputs);

    // !move result to host!
    for (size_t i = 0; i < outputs.size(); i++)
    {
        outputs[i].ToHost();
    }

    return APP_ERR_OK;
};

APP_ERROR FaceLandMark::PostProcess(std::vector<MxBase::Tensor> &outputs,
                                    const cv::Mat &inputMat,
                                    cv::Mat &affinedMat)
{
    cv::Mat data =
        cv::Mat(cv::Size(LANDMARK_OUTPUT_SIZE * LANDMARK_OUTPUT_SIZE, LANDMARK_NUMS), CV_32FC1, (float32_t *)outputs[1].GetData());

    double maxValue[LANDMARK_NUMS] = {0};
    std::vector<cv::Point2f> landmarkPoints;
    for (size_t i = 0; i < LANDMARK_NUMS; i++)
    {
        landmarkPoints.push_back(cv::Point());
    }

    // find the most possible landmark points
    for (size_t h = 0; h < LANDMARK_NUMS; h++)
    {
        for (size_t w = 0; w < LANDMARK_OUTPUT_SIZE * LANDMARK_OUTPUT_SIZE; w++)
        {
            auto values = data.at<cv::Vec<float32_t, 1>>(h, w);
            if (values[h] > maxValue[h])
            {
                maxValue[h] = values[0];
                landmarkPoints[h] = cv::Point2f(w % LANDMARK_OUTPUT_SIZE, w / LANDMARK_OUTPUT_SIZE);
            }
        }
    }

    // resize for affine transform
    cv::Mat resizedMat;
    cv::resize(inputMat, resizedMat, cv::Size(AFFINE_SIZE, AFFINE_SIZE));

    for (auto &point : landmarkPoints)
    {
        point.x *= ((double_t)AFFINE_SIZE / LANDMARK_OUTPUT_SIZE);
        point.y *= ((double_t)AFFINE_SIZE / LANDMARK_OUTPUT_SIZE);
    }

    // do affine transform
    cv::Mat affineParam = cv::estimateAffinePartial2D(landmarkPoints, TARGET_POINTS);
    cv::warpAffine(resizedMat, affinedMat, affineParam, cv::Size(AFFINE_SIZE, AFFINE_SIZE));

    return APP_ERR_OK;
};