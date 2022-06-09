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
#include "MxpiAlphaposePreProcess.h"
#include <numeric>
#include <algorithm>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include <string>

using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
using namespace cv;

namespace {
    const int MODEL_WIDTH = 192;
    const int MODEL_HEIGHT = 256;
    const int CENTERX_INDEX = 0;
    const int CENTERY_INDEX = 1;
    const int SCALEW_INDEX = 2;
    const int SCALEH_INDEX = 3;
    const float HALF = 0.5;
    bool ACC_TEST = false;
}

/**
 * @brief Get decoded image and change it from yuvNV12 to RGB
 * @param srcMxpiVisionList - Source srcMxpiVisionList
 * @param decodedImage - Decoded RGB image
 */
static void GetDecodedImages(const MxTools::MxpiVisionList srcMxpiVisionList, cv::Mat &decodedImage)
{
    if (ACC_TEST) {
        MxpiVisionData inputdata = srcMxpiVisionList.visionvec(0).visiondata();
        cv::Mat rawData(1, (uint32_t)inputdata.datasize(), CV_8UC1, (void *)inputdata.dataptr());
        decodedImage = cv::imdecode(rawData, cv::IMREAD_COLOR);
    } else {
        // Get decoded image from image decoder
        MxTools::MxpiVision srcMxpiVision = srcMxpiVisionList.visionvec(0);
        // Copy memory from device to host
        MxBase::MemoryData dstHost((uint32_t)srcMxpiVision.visiondata().datasize(), MxBase::MemoryData::MEMORY_HOST);
        MxBase::MemoryData srcDvpp((void *)srcMxpiVision.visiondata().dataptr(), (uint32_t)srcMxpiVision.visiondata().datasize(),
                                   MxBase::MemoryData::MEMORY_DVPP);
        MemoryHelper::MxbsMallocAndCopy(dstHost, srcDvpp);

        // yuv --> bgr
        int yuvBytesMu = 3;
        int yuvBytesNu = 2;
        int height = srcMxpiVision.visioninfo().heightaligned();
        int width = srcMxpiVision.visioninfo().widthaligned();
        cv::Mat yuvImage(height * yuvBytesMu / yuvBytesNu, width, CV_8UC1, Scalar(0));
        memcpy(yuvImage.data, static_cast<unsigned char*>(dstHost.ptrData), width * height * yuvBytesMu / yuvBytesNu * sizeof(unsigned char));
        Mat rgbImg(height, width, CV_8UC3, Scalar(0, 0, 0));
        cv::cvtColor(yuvImage, rgbImg, COLOR_YUV2BGR_NV12);
        if (rgbImg.isContinuous()) {
            decodedImage = rgbImg;
        } else {
            decodedImage = rgbImg.clone();
        }
        dstHost.free(dstHost.ptrData);
    }
}

/**
 * @brief decode MxpiObjectList
 * @param srcMxpiObjectList - Source MxpiObjectList
 * @param objectBoxes - The boxes of object
 */
static void GetBoxes(const MxTools::MxpiObjectList srcMxpiObjectList,
                     std::vector<std::vector<float> > &objectBoxes)
{
    int boxInfoNum = 4;
    float scaleMult = 1.25;
    float aspectRatio = 0.75;
    for (int i = 0; i < srcMxpiObjectList.objectvec_size(); i++) {
        MxTools::MxpiObject srcMxpiObject = srcMxpiObjectList.objectvec(i);
        // Filter out person class
        if ((ACC_TEST) || (srcMxpiObject.classvec(0).classid() == 0)) {
            std::vector<float> objectBox(boxInfoNum);
            float x0 = srcMxpiObject.x0();
            float y0 = srcMxpiObject.y0();
            float x1 = srcMxpiObject.x1();
            float y1 = srcMxpiObject.y1();
            float centerx = (x1 + x0) * HALF;
            float centery = (y1 + y0) * HALF;
            float boxWidth = x1 - x0;
            float boxHeight = y1 - y0;
            // Adjust the aspect ratio
            if (boxWidth >= aspectRatio * boxHeight) {
                boxHeight = boxWidth / aspectRatio;
            } else {
                boxWidth = boxHeight * aspectRatio;
            }
            float scalew = boxWidth * scaleMult;
            float scaleh = boxHeight * scaleMult;

            objectBox[CENTERX_INDEX] = centerx;
            objectBox[CENTERY_INDEX] = centery;
            objectBox[SCALEW_INDEX] = scalew;
            objectBox[SCALEH_INDEX] = scaleh;
            objectBoxes.push_back(objectBox);
        }
    }
}

/**
 * @brief Get the third mapPoint for affine transform
 * @param mapPoint - cv::Point2f *
 */
static void GetThirdPoint(cv::Point2f *mapPoint)
{
    int thirdpointIndex = 2;
    float directx = mapPoint[0].x - mapPoint[1].x;
    float directy = mapPoint[0].y - mapPoint[1].y;
    mapPoint[thirdpointIndex].x = mapPoint[1].x - directy;
    mapPoint[thirdpointIndex].y = mapPoint[1].y + directx;
}

/**
 * @brief Get the transformation matrix for affine tranform
 * @param center - The center of object box
 * @param scale - The scale of objetc box
 * @param outputSize - The transformation matrix for affine tranform
 * @param trans - The transformation matrix for affine tranform
 */
static void GetAffineTransform(const std::vector<float> &center, const std::vector<float> &scale,
                               const std::vector<int> &outputSize, cv::Mat &trans)
{
    int pointNum = 3;
    cv::Point2f src[pointNum];
    src[0].x = center[0];
    src[0].y = center[1];
    src[1].x = center[0];
    src[1].y = center[1] - scale[0] * HALF;
    GetThirdPoint(src);
    cv::Point2f dst[pointNum];
    dst[0].x = outputSize[0] * HALF;
    dst[0].y = outputSize[1] * HALF;
    dst[1].x = outputSize[0] * HALF;
    dst[1].y = (outputSize[1] - outputSize[0]) * HALF;
    GetThirdPoint(dst);
    trans = cv::getAffineTransform(src, dst);
}

/**
 * @brief Affine transformation
 * @param decodedImage - Decoded RGB image
 * @param objectBoxes - The boxes of object
 * @param affinedImages - The image after affine transformation
 */
static void DoWarpAffine(const cv::Mat &decodedImage,
                         const std::vector<std::vector<float> > &objectBoxes,
                         std::vector<cv::Mat> &affinedImages)
{
    std::vector<int> outputSize = {MODEL_WIDTH, MODEL_HEIGHT};
    int batchSize = objectBoxes.size();
    for (int i = 0; i < batchSize; i++) {
        std::vector<float> center = {};
        center.push_back(objectBoxes[i][CENTERX_INDEX]);
        center.push_back(objectBoxes[i][CENTERY_INDEX]);
        std::vector<float> scale = {};
        scale.push_back(objectBoxes[i][SCALEW_INDEX]);
        scale.push_back(objectBoxes[i][SCALEH_INDEX]);
        int transxIndex = 3;
        int transyIndex = 2;
        cv::Mat trans(transyIndex, transxIndex, CV_32FC1, Scalar(0));
        // Get transformation matrix for affine transformation
        GetAffineTransform(center, scale, outputSize, trans);
        cv::Mat dst(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3, Scalar(0, 0, 0));
        // Affine transformation
        cv::warpAffine(decodedImage, dst, trans, dst.size());

        if (dst.isContinuous()) {
            affinedImages.push_back(dst);
        } else {
            affinedImages.push_back(dst.clone());
        }
    }
}

/**
 * @brief Prepare output in the format of MxpiVisionList
 * @param affinedImages - The image after affine transformation
 * @param dstMxpiVisionList - Target data in the format of MxpiVisionList
 * @return APP_ERROR
 */
APP_ERROR MxpiAlphaposePreProcess::GenerateMxpiOutput(std::vector<cv::Mat> &affinedImages,
                                                      MxpiVisionList &dstMxpiVisionList)
{
    int rgbSize = 3;
    for (int i = 0; i < affinedImages.size(); i++) {
        auto mxpiVisionPtr = dstMxpiVisionList.add_visionvec();
        // Set vision infomation
        mxpiVisionPtr->mutable_visioninfo()->set_width((uint32_t)MODEL_WIDTH);
        mxpiVisionPtr->mutable_visioninfo()->set_height((uint32_t)MODEL_HEIGHT);
        mxpiVisionPtr->mutable_visioninfo()->set_widthaligned((uint32_t)MODEL_WIDTH);
        mxpiVisionPtr->mutable_visioninfo()->set_heightaligned((uint32_t)MODEL_HEIGHT);

        // Copy memmory from host to device
        MxBase::MemoryData srcImage((void *)affinedImages[i].data, (uint32_t)(MODEL_WIDTH * MODEL_HEIGHT * rgbSize),
                                    MxBase::MemoryData::MEMORY_HOST);
        MxBase::MemoryData dstImage((uint32_t)(MODEL_WIDTH * MODEL_HEIGHT * rgbSize), MxBase::MemoryData::MEMORY_DEVICE);
        MemoryHelper::MxbsMallocAndCopy(dstImage, srcImage);
        // Set vision data
        mxpiVisionPtr->mutable_visiondata()->set_dataptr((const uint64_t)dstImage.ptrData);
        std::string str = (const char *)dstImage.ptrData;
        mxpiVisionPtr->mutable_visiondata()->set_datastr(str);
        mxpiVisionPtr->mutable_visiondata()->set_datasize(MODEL_WIDTH * MODEL_HEIGHT * rgbSize);
        mxpiVisionPtr->mutable_visiondata()->set_deviceid(0);
        mxpiVisionPtr->mutable_visiondata()->set_memtype(MxTools::MxpiMemoryType::MXPI_MEMORY_DEVICE);
        mxpiVisionPtr->mutable_visiondata()->set_freefunc(0);
        MxTools::MxpiMetaHeader *header = mxpiVisionPtr->add_headervec();
        header->set_datasource(parentName_);
        header->set_memberid(i);
    }
    return APP_ERR_OK;
}

/**
 * @brief Overall process to generate pretreated images
 * @param srcMxpiObjectList - Source MxpiObjectList containing object data about input image
 * @param srcMxpiVisionList - Source MxpiTensorPackageList containing input image
 * @param dstMxpiVisionList - Target MxpiVisionList containing detection result list
 * @return APP_ERROR
 */
APP_ERROR MxpiAlphaposePreProcess::GenerateVisionList(const MxpiObjectList &srcMxpiObjectList,
                                                      const MxpiVisionList &srcMxpiVisionList,
                                                      MxpiVisionList &dstMxpiVisionList)
{
    // Get object boxes from object detector
    std::vector<std::vector<float> > objectBoxes = {};
    GetBoxes(srcMxpiObjectList, objectBoxes);
    std::vector<cv::Mat> affinedImages = {};
    if (objectBoxes.size() == 0) {
        LogWarn << "There is no people in this frame or picture";
        cv::Mat affinedImage(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3, Scalar(0));
        affinedImages.push_back(affinedImage);
    } else {
        // Get images from image decoder
        cv::Mat decodedImage;
        GetDecodedImages(srcMxpiVisionList, decodedImage);
        // Do affine transform
        DoWarpAffine(decodedImage, objectBoxes, affinedImages);
    }

    // Prepare output in the format of MxpiVisionList
    GenerateMxpiOutput(affinedImages, dstMxpiVisionList);
    return APP_ERR_OK;
}

/**
 * @brief Initialize configure parameter.
 * @param configParamMap
 * @return APP_ERROR
 */
APP_ERROR MxpiAlphaposePreProcess::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogInfo << "MxpiAlphaposePreProcess::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    std::shared_ptr<string> imageDecoderPropSptr = std::static_pointer_cast<string>(configParamMap["imageSource"]);
    imageDecoderName_ = *imageDecoderPropSptr.get();
    return APP_ERR_OK;
}

/**
 * @brief DeInitialize configure parameter.
 * @return APP_ERROR
 */
APP_ERROR MxpiAlphaposePreProcess::DeInit()
{
    LogInfo << "MxpiAlphaposePreProcess::DeInit end.";
    LogInfo << "MxpiAlphaposePreProcess::DeInit end.";
    return APP_ERR_OK;
}

/**
 * @brief Process the data of MxpiBuffer.
 * @param mxpiBuffer
 * @return APP_ERROR
 */
APP_ERROR MxpiAlphaposePreProcess::Process(std::vector<MxpiBuffer*> &mxpiBuffer)
{
    MxpiBuffer *buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiBufferManager mxpiBufferManager;
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) <<
        "MxpiAlphaposePreProcess process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiAlphaposePreProcess process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the output of objectpostprocessor from buffer
    shared_ptr<void> objectMetadata = mxpiMetadataManager.GetMetadata(parentName_);
    if (objectMetadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "object metadata is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_METADATA_IS_NULL;
    }
    shared_ptr<MxpiObjectList> srcMxpiObjectListSptr
	    = static_pointer_cast<MxpiObjectList>(objectMetadata);
    MxpiObjectList srcMxpiObjectList = *srcMxpiObjectListSptr;

    MxpiVisionList srcMxpiVisionList;
    if (imageDecoderName_ == "appInput") {
        ACC_TEST = true;
        srcMxpiVisionList = mxpiBufferManager.GetHostDataInfo(*buffer).visionlist();
    } else {
        // Get the output of objectdetector from buffer
        shared_ptr<void> imageMetadata = mxpiMetadataManager.GetMetadata(imageDecoderName_);
        if (imageMetadata == nullptr) {
            ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "imageDecoder metadata is NULL, failed";
            mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
            mxpiErrorInfo.errorInfo = ErrorInfo_.str();
            SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
            return APP_ERR_METADATA_IS_NULL;
        }
        shared_ptr<MxpiVisionList> srcMxpiVisionListSptr
            = static_pointer_cast<MxpiVisionList>(imageMetadata);
        srcMxpiVisionList = *srcMxpiVisionListSptr;
    }
    // Generate output
    shared_ptr<MxpiVisionList> dstMxpiVisionListSptr =
            make_shared<MxpiVisionList>();
    APP_ERROR ret = GenerateVisionList(srcMxpiObjectList, srcMxpiVisionList, *dstMxpiVisionListSptr);
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiAlphaposePreProcess get person's keypoint information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiVisionListSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiAlphaposePreProcess add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    return APP_ERR_OK;
}


/**
 * @brief Definition the parameter of configure properties.
 * @return std::vector<std::shared_ptr<void>>
 */
std::vector<std::shared_ptr<void>> MxpiAlphaposePreProcess::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parentNameProSptr = (std::make_shared<ElementProperty<string>>)(ElementProperty<string> {
            STRING, "dataSource", "parentName", "the name of previous plugin", "mxpi_objectpostprocessor0", "NULL", "NULL"});
    auto imageDecoderProSptr = (std::make_shared<ElementProperty<string>>)(ElementProperty<string> {
            STRING, "imageSource", "imageDecoderName", "the name of image decoder plugin", "mxpi_imagedecoder0", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    properties.push_back(imageDecoderProSptr);

    return properties;
}

APP_ERROR MxpiAlphaposePreProcess::SetMxpiErrorInfo(MxpiBuffer &buffer, const std::string plugin_name,
                                                    const MxpiErrorInfo mxpiErrorInfo)
{
    APP_ERROR ret = APP_ERR_OK;
    // Define an object of MxpiMetadataManager
    MxpiMetadataManager mxpiMetadataManager(buffer);
    ret = mxpiMetadataManager.AddErrorInfo(plugin_name, mxpiErrorInfo);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to AddErrorInfo.";
        return ret;
    }
    ret = SendData(0, buffer);
    return ret;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(MxpiAlphaposePreProcess)