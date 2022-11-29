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
#include "RetinafaceDetection.h"

#include <sys/stat.h>
#include <unistd.h>

#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Log/Log.h"
#include "boost/filesystem.hpp"
#include "opencv2/opencv.hpp"
namespace fs = boost::filesystem;
using namespace MxBase;

namespace {
const uint32_t IMAGE_SIZE = 1000;
const int NETINPUTSIZE = 1000;
std::string imagePath;
int originImageW;
int originImageH;
float resize;
int padLeft;
int padRight;
int padTop;
int padBottom;
const uint32_t YUV_BYTE_NU = 3;
const uint32_t YUV_BYTE_DE = 2;
const uint32_t VPC_H_ALIGN = 2;
}  // namespace
void RetinafaceDetection::SetRetinafacePostProcessConfig(
    const InitParam& initParam,
    std::map<std::string, std::shared_ptr<void>>& config) {
  MxBase::ConfigData configData;
  const std::string checkTensor = initParam.checkTensor ? "true" : "false";
  configData.SetJsonValue("CHECK_MODEL", checkTensor);
  configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
  auto jsonStr = configData.GetCfgJson().serialize();
  config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
  config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);
}
APP_ERROR RetinafaceDetection::Init(const InitParam& initParam) {
  deviceId_ = initParam.deviceId;
  APP_ERROR ret = MxBase::DeviceManager::GetInstance()->InitDevices();
  if (ret != APP_ERR_OK) {
    LogError << "Init devices failed, ret=" << ret << ".";
    return ret;
  }
  ret = MxBase::TensorContext::GetInstance()->SetContext(initParam.deviceId);
  if (ret != APP_ERR_OK) {
    LogError << "Set context failed, ret=" << ret << ".";
    return ret;
  }
  dvppWrapper_ = std::make_shared<MxBase::DvppWrapper>();
  ret = dvppWrapper_->Init();
  if (ret != APP_ERR_OK) {
    LogError << "DvppWrapper init failed, ret=" << ret << ".";
    return ret;
  }
  model_ = std::make_shared<MxBase::ModelInferenceProcessor>();
  ret = model_->Init(initParam.modelPath, modelDesc_);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInferenceProcessor init failed, ret=" << ret << ".";
    return ret;
  }
  // init Retinafacepostprocess
  std::map<std::string, std::shared_ptr<void>> config;
  SetRetinafacePostProcessConfig(initParam, config);
  post_ = std::make_shared<RetinafacePostProcess>();
  cv::Mat originalImage = cv::imread(initParam.ImagePath);
  ret = post_->Init(config, originalImage.rows, originalImage.cols);
  if (ret != APP_ERR_OK) {
    LogError << "Retinafacepostprocess init failed, ret = " << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR RetinafaceDetection::DeInit() {
  dvppWrapper_->DeInit();
  model_->DeInit();
  post_->DeInit();
  MxBase::DeviceManager::GetInstance()->DestroyDevices();
  return APP_ERR_OK;
}

// 获取图像数据，将数据存入TensorBase中
APP_ERROR RetinafaceDetection::ReadImage(const std::string& imgPath,
                                         cv::Mat& imageMat) {
  imageMat = cv::imread(imgPath, cv::IMREAD_COLOR);
  return APP_ERR_OK;
}

APP_ERROR RetinafaceDetection::CVMatToTensorBase(
    const cv::Mat& imageMat, MxBase::TensorBase& tensorBase) {
  const uint32_t dataSize = imageMat.cols * imageMat.rows * YUV444_RGB_WIDTH_NU;
  LogInfo << "image size " << imageMat.cols << " " << imageMat.rows;
  MemoryData memoryDataDst(dataSize, MemoryData::MEMORY_DEVICE, deviceId_);
  MemoryData memoryDataSrc(imageMat.data, dataSize,
                           MemoryData::MEMORY_HOST_MALLOC);

  APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(memoryDataDst, memoryDataSrc);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret) << "Memory malloc failed.";
    return ret;
  }

  std::vector<uint32_t> shape = {imageMat.rows * YUV444_RGB_WIDTH_NU,
                                 static_cast<uint32_t>(imageMat.cols)};
  tensorBase = TensorBase(memoryDataDst, false, shape, TENSOR_DTYPE_UINT8);
  return APP_ERR_OK;
}

// 模型推理
APP_ERROR RetinafaceDetection::Inference(
    const std::vector<MxBase::TensorBase>& inputs,
    std::vector<MxBase::TensorBase>& outputs) {
  auto dtypes = model_->GetOutputDataType();
  for (size_t i = 0; i < modelDesc_.outputTensors.size(); ++i) {
    std::vector<uint32_t> shape = {};
    for (size_t j = 0; j < modelDesc_.outputTensors[i].tensorDims.size(); ++j) {
      shape.push_back((uint32_t)modelDesc_.outputTensors[i].tensorDims[j]);
    }
    TensorBase tensor(shape, dtypes[i], MemoryData::MemoryType::MEMORY_DEVICE,
                      deviceId_);
    APP_ERROR ret = TensorBase::TensorBaseMalloc(tensor);
    if (ret != APP_ERR_OK) {
      LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
      return ret;
    }
    outputs.push_back(tensor);
  }
  // print the shape and type of inputs
  std::cout << "inputs size = " << inputs[0].GetSize() << "\n";
  std::cout << "inputs GetByteSize = " << inputs[0].GetByteSize() << "\n";
  std::cout << "inputs number = " << inputs.size() << "\n";
  std::cout << "inputs shape size = " << inputs[0].GetShape().size() << "\n";
  std::cout << "Data type = " << inputs[0].GetDataType() << "\n";
  for (size_t i = 0; i < inputs[0].GetShape().size(); i++) {
    std::cout << "value = ";
    std::cout << inputs[0].GetShape()[i] << " ";
  }
  DynamicInfo dynamicInfo = {};
  dynamicInfo.dynamicType = DynamicType::STATIC_BATCH;
  LogInfo << "Ready to infer.";
  APP_ERROR ret = model_->ModelInference(inputs, outputs, dynamicInfo);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInference failed, ret=" << ret << ".";
    return ret;
  }
  LogInfo << "End to model inference.";
  return APP_ERR_OK;
}

// 后处理
APP_ERROR RetinafaceDetection::PostProcess(
    const std::vector<MxBase::TensorBase>& outputs,
    std::vector<std::vector<MxBase::ObjectInfo>>& objInfos) {
  LogInfo << "start postprocess.\n";
  APP_ERROR ret = post_->Process(outputs, objInfos);
  if (ret != APP_ERR_OK) {
    LogError << "Process failed, ret=" << ret << ".";
    return ret;
  }

  ret = post_->DeInit();
  if (ret != APP_ERR_OK) {
    LogError << "RetinafacePostProcess DeInit failed";
    return ret;
  }
  LogInfo << "End to Retinafacepostprocess.";
  return APP_ERR_OK;
}

APP_ERROR RetinafaceDetection::WriteResult(
    const std::string& imgPath,
    const std::vector<std::vector<MxBase::ObjectInfo>>& objInfos) {
  LogInfo << "start write result.";
  cv::Mat writeImage = cv::imread(imagePath);
  uint32_t objInfosSize = objInfos.size();
  std::vector<MxBase::ObjectInfo> resultInfo;
  std::cout << "objInfo number = " << objInfosSize << std::endl;
  for (uint32_t i = 0; i < objInfosSize; i++) {
    for (uint32_t j = 0; j < objInfos[i].size(); j++) {
      resultInfo.push_back(objInfos[i][j]);
    }
    LogInfo << "result box number is : " << resultInfo.size();
    for (uint32_t j = 0; j < resultInfo.size(); j++) {
      const uint32_t thickness = 2;
      const cv::Scalar black = cv::Scalar(0, 0, 0);
      int X0 = std::max((int)((resultInfo[j].x0 - padLeft) / resize), 0);
      int X1 = std::max((int)((resultInfo[j].x1 - padLeft) / resize), 0);
      int Y0 = std::max((int)((resultInfo[j].y0 - padTop) / resize), 0);
      int Y1 = std::max((int)((resultInfo[j].y1 - padTop) / resize), 0);
      cv::Point2i c1(X0, Y0);
      cv::Point2i c2(X1, Y1);
      cv::rectangle(writeImage, cv::Rect(X0, Y0, X1 - X0, Y1 - Y0), black,
                    thickness);
    }
  }
  cv::imwrite("./result.jpg", writeImage);
  return APP_ERR_OK;
}

APP_ERROR RetinafaceDetection::Process(const std::string& imgPath) {
  imagePath = imgPath;
  cv::Mat originImage = cv::imread(imgPath);
  if (originImage.data == NULL) {
    LogInfo << "The image is not exist.\n";
    return 0;
  }
  originImageW = originImage.cols;
  originImageH = originImage.rows;
  int imgsizeMax = originImageW;
  if (imgsizeMax < originImageH) {
    imgsizeMax = originImageH;
  }
  resize = (IMAGE_SIZE * 1.0) / (imgsizeMax * 1.0);
  cv::Mat newImg;
  cv::resize(originImage, newImg, cv::Size(), resize, resize,
             cv::INTER_NEAREST);
  padRight = IMAGE_SIZE - newImg.cols;
  padLeft = 0;
  padBottom = IMAGE_SIZE - newImg.rows;
  padTop = 0;
  cv::Mat nnImage;
  cv::copyMakeBorder(newImg, nnImage, padTop, padBottom, padLeft, padRight,
                     cv::BORDER_CONSTANT, 0);
  std::cout << "nnImage W = " << nnImage.cols << " "
            << "nnImage H = " << nnImage.rows << "\n";
  std::string newImagePath = "./ImageforInfer.jpg";
  cv::imwrite(newImagePath, nnImage);
  cv::Mat imageMat;
  APP_ERROR ret = ReadImage(newImagePath, imageMat);
  if (ret != APP_ERR_OK) {
    LogError << "ReadImage failed, ret=" << ret << ".";
    return ret;
  }
  std::vector<MxBase::TensorBase> inputs = {};
  std::vector<MxBase::TensorBase> outputs = {};
  TensorBase tensorBase;
  ret = CVMatToTensorBase(imageMat, tensorBase);
  if (ret != APP_ERR_OK) {
    LogError << "CVMatToTensorBase failed, ret=" << ret << ".";
    return ret;
  }
  inputs.push_back(tensorBase);

  ret = Inference(inputs, outputs);
  if (ret != APP_ERR_OK) {
    LogError << "Inference failed, ret=" << ret << ".";
    return ret;
  }

  std::vector<std::vector<MxBase::ObjectInfo>> objInfos;
  std::cout << std::endl;
  std::cout << "outputSize = " << outputs.size() << std::endl;
  for (uint32_t i = 0; i < outputs.size(); i++) {
    for (uint32_t j = 0; j < outputs[i].GetShape().size(); j++) {
      std::printf("outputs[%d][%d] = %d. ", i, j, outputs[i].GetShape()[j]);
    }
    std::cout << std::endl;
  }
  ret = PostProcess(outputs, objInfos);
  if (ret != APP_ERR_OK) {
    LogError << "PostProcess failed, ret=" << ret << ".";
    return ret;
  }

  ret = WriteResult(imgPath, objInfos);
  if (ret != APP_ERR_OK) {
    LogError << "Save result failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}