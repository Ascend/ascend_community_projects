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

#include "FCOSDetectionPostProcess.h"

#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"

namespace {
const float RATE = 0.3;
const float MINRATE = 0.56;
const uint32_t L = 0;
const uint32_t T = 1;
const uint32_t R = 2;
const uint32_t B = 3;
const int NETINPUTWIDTH = 1333;
const int NETINPUTHEIGHT = 800;
const uint32_t CENTERPOINT = 4;
const float THRESHOLD_ = 0.3;
}  // namespace
using namespace MxBase;

FCOSPostProcess &FCOSPostProcess::operator=(const FCOSPostProcess &other) {
  if (this == &other) {
    return *this;
  }
  ObjectPostProcessBase::operator=(other);
  return *this;
}

APP_ERROR FCOSPostProcess::Init(
    const std::map<std::string, std::shared_ptr<void>> &postConfig) {
  LogInfo << "Start to Init FCOSDetectionPostProcess";
  APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
  if (ret != APP_ERR_OK) {
    LogError << GetError(ret)
             << "Fail to superInit in FCOSDetectionPostProcess.";
    return ret;
  }
  LogInfo << "End to Init FCOSDetectionPostProcess.";
  return APP_ERR_OK;
}

APP_ERROR FCOSPostProcess::DeInit() { return APP_ERR_OK; }

/*
    input:
        tensors:the output of mxpi_tensorinfer0 , the output of the model.
        objectInfos:save result.
    return:
        return the postprocess result.
*/
APP_ERROR FCOSPostProcess::Process(
    const std::vector<TensorBase> &tensors,
    std::vector<std::vector<ObjectInfo>> &objectInfos,
    const std::vector<ResizedImageInfo> &resizedImageInfos,
    const std::map<std::string, std::shared_ptr<void>> &configParamMap) {
  LogInfo << "Start to Process FCOSDetectionPostProcess.";
  APP_ERROR ret = APP_ERR_OK;
  auto inputs = tensors;
  ret = CheckAndMoveTensors(inputs);
  if (ret != APP_ERR_OK) {
    LogError << "CheckAndMoveTensors failed. ret=" << ret;
    return ret;
  }

  LogInfo << "FCOSDetectionPostProcess start to write results.";

  for (auto num : {0, 1}) {
    if (((uint32_t)num >= (uint32_t)tensors.size()) || (num < 0)) {
      LogError << GetError(APP_ERR_INVALID_PARAM) << "TENSOR(" << num
               << ") must ben less than tensors'size(" << tensors.size()
               << ") and larger than 0.";
    }
  }
  auto shape = tensors[0].GetShape();
  if (shape.size() == 0) {
    return APP_ERR_OK;
  }
  LogInfo << "start to process.";
  if (tensors[0].GetBuffer() == NULL || tensors[1].GetBuffer() == NULL) {
    LogError << "tensors buffer is NULL.\n";
    return APP_ERR_OK;
  }
  std::vector<ObjectInfo> objectInfo;
  auto res0 = (float *)tensors[0].GetBuffer();
  auto classIdx = (__int64 *)tensors[1].GetBuffer();
  auto shape1 = tensors[1].GetShape();
  for (uint32_t i = 0; i < shape[1]; i++) {
    float *beginRes = res0 + i * 5;
    if (*(beginRes + CENTERPOINT) >= THRESHOLD_) {
      ObjectInfo objInfo;
      objInfo.x0 = *(beginRes + L);
      objInfo.y0 = *(beginRes + T);
      objInfo.x1 = *(beginRes + R);
      objInfo.y1 = *(beginRes + B);
      objInfo.confidence = *(beginRes + CENTERPOINT);
      objInfo.classId = (float)classIdx[i];
      LogInfo << "start postprocessbbox.";
      PostprocessBBox(objInfo, resizedImageInfos[0].widthOriginal,
                      resizedImageInfos[0].heightOriginal, NETINPUTWIDTH,
                      NETINPUTHEIGHT);
      objectInfo.push_back(objInfo);
    }
  }
  MxBase::NmsSort(objectInfo, RATE);
  objectInfos.push_back(objectInfo);
  LogInfo << "FCOSDetectionPostProcess write results successed.";
  LogInfo << "End to Process FCOSDetectionPostProcess.";
  return APP_ERR_OK;
}

void FCOSPostProcess::PostprocessBBox(ObjectInfo &objInfo, int imageWidth,
                                      int imageHeight, int netInWidth,
                                      int netInHeight) {
  float scale = netInWidth * 1.0 / imageWidth * 1.0;
  if (scale > (netInHeight * 1.0 / imageHeight * 1.0)) {
    scale = (netInHeight * 1.0 / imageHeight * 1.0);
  }
  float padW = netInWidth * 1.0 - imageWidth * 1.0 * scale;
  float padH = netInHeight * 1.0 - imageHeight * 1.0 * scale;
  float padLeft = padW / 2;
  float padTop = padH / 2;
  objInfo.x0 = (objInfo.x0 - padLeft) / scale;
  objInfo.y0 = (objInfo.y0 - padTop) / scale;
  objInfo.x1 = (objInfo.x1 - padLeft) / scale;
  objInfo.y1 = (objInfo.y1 - padTop) / scale;
}