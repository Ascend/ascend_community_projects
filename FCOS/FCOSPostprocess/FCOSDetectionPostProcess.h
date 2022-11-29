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

#ifndef FCOS_POST_PROCESS_H
#define FCOS_POST_PROCESS_H
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "opencv2/opencv.hpp"

class FCOSPostProcess : public MxBase::ObjectPostProcessBase {
public:
  FCOSPostProcess() = default;
  ~FCOSPostProcess() = default;
  FCOSPostProcess(const FCOSPostProcess &other);
  APP_ERROR Init(
      const std::map<std::string, std::shared_ptr<void>> &postConfig) override;
  APP_ERROR DeInit() override;
  APP_ERROR Process(
      const std::vector<MxBase::TensorBase> &tensors,
      std::vector<std::vector<MxBase::ObjectInfo>> &objectInfos,
      const std::vector<MxBase::ResizedImageInfo> &resizedImageInfos = {},
      const std::map<std::string, std::shared_ptr<void>> &configParamMap = {})
      override;
  FCOSPostProcess &operator=(const FCOSPostProcess &other);

protected:
  void PostprocessBBox(MxBase::ObjectInfo &objInfo, int imageWidth,
                       int imageHeight, int netInWidth, int netInHeight);

private:
  uint32_t classNum_ = 0;
  bool softmax_ = true;
  uint32_t topK_ = 1;
  float min_confidence = 0.5;
};
#endif