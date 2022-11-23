#include "FCOSDetection.h"

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
const int RESTENSORF[2] = {100, 5};
const int RESTENSORS[2] = {100, 1};
const int NETINPUTWIDTH = 1333;
const int NETINPUTHEIGHT = 800;
const uint32_t VPC_H_ALIGN = 2;
const uint32_t YUV_BYTE_NU = 3;
const uint32_t YUV_BYTE_DE = 2;
std::string imagePath;
int originImageW;
int originImageH;
float scaleRatio;
int padLeft;
int padRight;
int padTop;
int padBottom;
}  // namespace
// load label file.
APP_ERROR FCOSDetection::LoadLabels(const std::string &labelPath,
                                    std::map<int, std::string> &labelMap) {
  std::ifstream infile;
  // open label file
  infile.open(labelPath, std::ios_base::in);
  std::string s;
  // check label file validity
  if (infile.fail()) {
    LogError << "Failed to open label file: " << labelPath << ".";
    return APP_ERR_COMM_OPEN_FAIL;
  }
  labelMap.clear();
  // construct label map
  int count = 0;
  while (std::getline(infile, s)) {
    if (s.find('#') <= 1) {
      continue;
    }
    size_t eraseIndex = s.find_last_not_of("\r\n\t");
    if (eraseIndex != std::string::npos) {
      s.erase(eraseIndex + 1, s.size() - eraseIndex);
    }
    labelMap.insert(std::pair<int, std::string>(count, s));
    count++;
  }
  infile.close();
  return APP_ERR_OK;
}

// Set model configuration parameters.
void FCOSDetection::SetFCOSPostProcessConfig(
    const InitParam &initParam,
    std::map<std::string, std::shared_ptr<void>> &config) {
  MxBase::ConfigData configData;
  const std::string checkTensor = initParam.checkTensor ? "true" : "false";
  configData.SetJsonValue("CLASS_NUM", std::to_string(initParam.classNum));
  configData.SetJsonValue("INPUT_TYPE", std::to_string(initParam.inputType));
  configData.SetJsonValue("CHECK_MODEL", checkTensor);
  auto jsonStr = configData.GetCfgJson().serialize();
  config["postProcessConfigContent"] = std::make_shared<std::string>(jsonStr);
  config["labelPath"] = std::make_shared<std::string>(initParam.labelPath);
}

APP_ERROR FCOSDetection::Init(const InitParam &initParam) {
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

  std::map<std::string, std::shared_ptr<void>> config;
  SetFCOSPostProcessConfig(initParam, config);
  // init FCOSPostprocess
  post_ = std::make_shared<FCOSPostProcess>();
  ret = post_->Init(config);
  if (ret != APP_ERR_OK) {
    LogError << "FCOSPostprocess init failed, ret=" << ret << ".";
    return ret;
  }
  // load labels from file
  ret = LoadLabels(initParam.labelPath, labelMap_);
  if (ret != APP_ERR_OK) {
    LogError << "Failed to load labels, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR FCOSDetection::DeInit() {
  dvppWrapper_->DeInit();
  model_->DeInit();
  post_->DeInit();
  MxBase::DeviceManager::GetInstance()->DestroyDevices();
  return APP_ERR_OK;
}

// get the image and send data to TensorBase.
APP_ERROR FCOSDetection::ReadImage(const std::string &imgPath,
                                   MxBase::TensorBase &tensor) {
  MxBase::DvppDataInfo inputDataInfo = {};
  MxBase::DvppDataInfo output = {};
  std::ifstream file(imgPath, std::ios::binary);
  // decode the image.
  if (!file) {
    LogError << "Invalid file.";
  }
  long fileSize = fs::file_size(imgPath);
  std::vector<char> buffer;
  buffer.resize(fileSize);
  file.read(buffer.data(), fileSize);
  file.close();
  std::string fileStr(buffer.data(), fileSize);
  MxBase::MemoryData hostMemory((void *)fileStr.c_str(), (size_t)fileStr.size(),
                                MemoryData::MEMORY_HOST, 0);
  MxBase::MemoryData dvppMemory(nullptr, (size_t)fileStr.size(),
                                MemoryData::MEMORY_DVPP, 0);
  APP_ERROR ret = MemoryHelper::MxbsMallocAndCopy(dvppMemory, hostMemory);
  ret = dvppWrapper_->DvppJpegPredictDecSize(
      hostMemory.ptrData, hostMemory.size, inputDataInfo.format,
      output.dataSize);

  inputDataInfo.dataSize = dvppMemory.size;
  inputDataInfo.data = (uint8_t *)dvppMemory.ptrData;
  ret = dvppWrapper_->DvppJpegDecode(inputDataInfo, output);
  ret = MemoryHelper::Free(dvppMemory);
  MxBase::MemoryData memoryData((void *)output.data, output.dataSize,
                                MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                                deviceId_);
  // judge the image size after decode.
  if (output.heightStride % VPC_H_ALIGN != 0) {
    LogError << "Output data height(" << output.heightStride
             << ") can't be divided by " << VPC_H_ALIGN << ".";
    MxBase::MemoryHelper::MxbsFree(memoryData);
    return APP_ERR_COMM_INVALID_PARAM;
  }
  std::vector<uint32_t> shape = {
      output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
  tensor =
      MxBase::TensorBase(memoryData, false, shape, MxBase::TENSOR_DTYPE_UINT8);
  return APP_ERR_OK;
}

APP_ERROR FCOSDetection::Resize(const MxBase::TensorBase &inputTensor,
                                MxBase::TensorBase &outputTensor) {
  auto shape = inputTensor.GetShape();
  MxBase::DvppDataInfo input = {};
  // Restore to original size.
  input.height = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
  input.width = shape[1];
  input.heightStride = (uint32_t)shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
  input.widthStride = shape[1];
  input.dataSize = inputTensor.GetByteSize();
  input.data = (uint8_t *)inputTensor.GetBuffer();
  const uint32_t resizeHeight = 800;
  const uint32_t resizeWidth = 1333;
  MxBase::ResizeConfig resize = {};
  resize.height = resizeHeight;
  resize.width = resizeWidth;
  MxBase::DvppDataInfo output = {};
  // resize image
  APP_ERROR ret = dvppWrapper_->VpcResize(input, output, resize);
  if (ret != APP_ERR_OK) {
    LogError << "VpcResize failed, ret=" << ret << ".";
    return ret;
  }
  MxBase::MemoryData memoryData((void *)output.data, output.dataSize,
                                MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                                deviceId_);
  // Determine the alignment size of the scaled image.
  if (output.heightStride % VPC_H_ALIGN != 0) {
    LogError << "Output data height(" << output.heightStride
             << ") can't be divided by " << VPC_H_ALIGN << ".";
    MxBase::MemoryHelper::MxbsFree(memoryData);
    return APP_ERR_COMM_INVALID_PARAM;
  }
  shape = {output.heightStride * YUV_BYTE_NU / YUV_BYTE_DE, output.widthStride};
  outputTensor =
      MxBase::TensorBase(memoryData, false, shape, MxBase::TENSOR_DTYPE_UINT8);
  return APP_ERR_OK;
}

// model reasoning
APP_ERROR FCOSDetection::Inference(
    const std::vector<MxBase::TensorBase> &inputs,
    std::vector<MxBase::TensorBase> &outputs) {
  auto dtypes = model_->GetOutputDataType();
  /* create room for result
   res_tensor[0] is 1*100*5
   res_tensor[1] is 1*100*1 */

  // create for res_tensor[0]
  std::vector<uint32_t> shape1 = {};
  shape1.push_back((uint32_t)RESTENSORF[0]);
  shape1.push_back((uint32_t)RESTENSORF[1]);
  MxBase::TensorBase tensor0(shape1, dtypes[0],
                             MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                             deviceId_);
  APP_ERROR ret = MxBase::TensorBase::TensorBaseMalloc(tensor0);
  if (ret != APP_ERR_OK) {
    LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
    return ret;
  }
  outputs.push_back(tensor0);

  // create for res_tensor[1]
  std::vector<uint32_t> shape2 = {};
  shape2.push_back((uint32_t)RESTENSORS[0]);
  shape2.push_back((uint32_t)RESTENSORS[1]);
  MxBase::TensorBase tensor1(shape2, dtypes[1],
                             MxBase::MemoryData::MemoryType::MEMORY_DEVICE,
                             deviceId_);
  ret = MxBase::TensorBase::TensorBaseMalloc(tensor1);
  if (ret != APP_ERR_OK) {
    LogError << "TensorBaseMalloc failed, ret=" << ret << ".";
    return ret;
  }
  outputs.push_back(tensor1);
  MxBase::DynamicInfo dynamicInfo = {};

  dynamicInfo.dynamicType = MxBase::DynamicType::STATIC_BATCH;
  ret = model_->ModelInference(inputs, outputs, dynamicInfo);
  if (ret != APP_ERR_OK) {
    LogError << "ModelInference failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}

// postprocess.
APP_ERROR FCOSDetection::PostProcess(
    const MxBase::TensorBase &tensor,
    const std::vector<MxBase::TensorBase> &outputs,
    std::vector<std::vector<MxBase::ObjectInfo>> &objInfos) {
  // save the resize information.
  auto shape = tensor.GetShape();
  MxBase::ResizedImageInfo imgInfo;
  imgInfo.widthOriginal = shape[1];
  imgInfo.heightOriginal = shape[0] * YUV_BYTE_DE / YUV_BYTE_NU;
  imgInfo.widthResize = NETINPUTWIDTH;
  imgInfo.heightResize = NETINPUTHEIGHT;
  imgInfo.resizeType = MxBase::RESIZER_STRETCHING;
  std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
  imageInfoVec.push_back(imgInfo);
  // use FCOSPostprocess.
  APP_ERROR ret = post_->Process(outputs, objInfos, imageInfoVec);
  if (ret != APP_ERR_OK) {
    LogError << "Process failed, ret=" << ret << ".";
    return ret;
  }

  ret = post_->DeInit();
  if (ret != APP_ERR_OK) {
    LogError << "FCOSPostprocess DeInit failed";
    return ret;
  }
  return APP_ERR_OK;
}

APP_ERROR FCOSDetection::WriteResult(
    MxBase::TensorBase &tensor,
    const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos) {
  APP_ERROR ret = tensor.ToHost();
  if (ret != APP_ERR_OK) {
    LogError << "ToHost faile";
    return ret;
  }
  auto shape = tensor.GetShape();
  cv::Mat imgBgr = cv::imread(imagePath);
  uint32_t batchSize = objInfos.size();
  std::vector<MxBase::ObjectInfo> resultInfo;
  for (uint32_t i = 0; i < batchSize; i++) {
    for (uint32_t j = 0; j < objInfos[i].size(); j++) {
      resultInfo.push_back(objInfos[i][j]);
    }
    // 打印置信度最大推理结果
    LogInfo << "result box number is : " << resultInfo.size();
    for (uint32_t j = 0; j < resultInfo.size(); j++) {
      const cv::Scalar green = cv::Scalar(0, 255, 0);
      const cv::Scalar black = cv::Scalar(0, 0, 0);
      const uint32_t thickness = 1;
      const uint32_t lineType = 8;
      const float fontScale = 1.0;

      int newX0 = std::max((int)((resultInfo[j].x0 - padLeft) / scaleRatio), 0);
      int newX1 = std::max((int)((resultInfo[j].x1 - padLeft) / scaleRatio), 0);
      int newY0 = std::max((int)((resultInfo[j].y0 - padTop) / scaleRatio), 0);
      int newY1 = std::max((int)((resultInfo[j].y1 - padTop) / scaleRatio), 0);
      int baseline = 0;
      const int WIDEBIAS = 15;
      const int HEIGHTBIAS = 3;
      const int YBIAS = 2;
      const float FONT = 3.0;
      std::string holdStr = std::to_string(resultInfo[j].confidence * 100.0);
      std::string confStr = holdStr.substr(0, holdStr.find(".") + 2 + 1);
      confStr = confStr + "% ";
      const uint32_t fontFace = cv::FONT_HERSHEY_SCRIPT_COMPLEX;
      cv::Point2i c1(newX0, newY0);
      cv::Point2i c2(newX1, newY1);
      cv::Size sSize = cv::getTextSize(confStr, fontFace, fontScale / 3,
                                       thickness, &baseline);
      cv::Size textSize =
          cv::getTextSize(labelMap_[((int)resultInfo[j].classId)], fontFace,
                          fontScale / FONT, thickness, &baseline);
      cv::rectangle(imgBgr, c1,
                    cv::Point(c1.x + textSize.width + WIDEBIAS + sSize.width,
                              c1.y - textSize.height - HEIGHTBIAS),
                    green, -1);
      // 在图像上绘制文字
      cv::putText(imgBgr,
                  labelMap_[((int)resultInfo[j].classId)] + ": " + confStr,
                  cv::Point(newX0, newY0 - YBIAS), cv::FONT_HERSHEY_SIMPLEX,
                  fontScale / FONT, black, thickness, lineType);
      // 绘制矩形
      cv::rectangle(imgBgr,
                    cv::Rect(newX0, newY0, newX1 - newX0, newY1 - newY0), green,
                    thickness);
    }
  }
  cv::imwrite("./result.jpg", imgBgr);
  return APP_ERR_OK;
}

APP_ERROR FCOSDetection::Process(const std::string &imgPath) {
  cv::Mat originImage = cv::imread(imgPath);
  if (originImage.data == NULL) {
    LogInfo << "The image is not exist.\n";
    return 0;
  }
  originImageW = originImage.cols;
  originImageH = originImage.rows;
  scaleRatio = (float)NETINPUTWIDTH * 1.0 / (originImageW * 1.0);
  float hold = (float)NETINPUTHEIGHT * 1.0 / (originImageH * 1.0);
  if (hold < scaleRatio) {
    scaleRatio = hold;
  }
  int newW = (int)originImageW * scaleRatio;
  int newH = (int)originImageH * scaleRatio;
  cv::Mat newImage;
  cv::resize(originImage, newImage, cv::Size(newW, newH), 0, 0, cv::INTER_AREA);
  const int PAD = 2;
  padLeft = std::max((int)((NETINPUTWIDTH - newW) / PAD), 0);
  padTop = std::max((int)((NETINPUTHEIGHT - newH) / PAD), 0);
  padRight = std::max(NETINPUTWIDTH - newW - padLeft, 0);
  padBottom = std::max(NETINPUTHEIGHT - newH - padTop, 0);
  cv::copyMakeBorder(newImage, newImage, padTop, padBottom, padLeft, padRight,
                     cv::BORDER_CONSTANT, 0);
  std::string newImagePath = "./ImageforInfer.jpg";
  cv::imwrite(newImagePath, newImage);
  MxBase::TensorBase inTensor;
  APP_ERROR ret = ReadImage(newImagePath, inTensor);
  imagePath = imgPath;
  if (ret != APP_ERR_OK) {
    LogError << "ReadImage failed, ret=" << ret << ".";
    return ret;
  }

  MxBase::TensorBase outTensor;
  ret = Resize(inTensor, outTensor);
  if (ret != APP_ERR_OK) {
    LogError << "Resize failed, ret=" << ret << ".";
    return ret;
  }
  std::vector<MxBase::TensorBase> inputs = {};
  std::vector<MxBase::TensorBase> outputs = {};
  inputs.push_back(outTensor);
  ret = Inference(inputs, outputs);
  if (ret != APP_ERR_OK) {
    LogError << "Inference failed, ret=" << ret << ".";
    return ret;
  }

  std::vector<std::vector<MxBase::ObjectInfo>> objInfos;
  ret = PostProcess(inTensor, outputs, objInfos);
  if (ret != APP_ERR_OK) {
    LogError << "PostProcess failed, ret=" << ret << ".";
    return ret;
  }

  ret = WriteResult(inTensor, objInfos);
  if (ret != APP_ERR_OK) {
    LogError << "Save result failed, ret=" << ret << ".";
    return ret;
  }
  return APP_ERR_OK;
}
