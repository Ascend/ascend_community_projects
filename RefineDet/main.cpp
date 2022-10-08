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

#include <iostream>
#include <vector>
#include <RefineDetDetection.h>
#include "MxBase/Log/Log.h"

namespace {
    const uint32_t CLASS_NU = 21;
    const uint32_t BIASES_NU = 18;
    const uint32_t ANCHOR_DIM = 3;
    const uint32_t YOLO_TYPE = 3;
}

void InitRefineDetParam(InitParam &initParam)
{
    initParam.deviceId = 0;
    initParam.labelPath = "./models/VOC.names";
    initParam.checkTensor = true;
    initParam.modelPath = "./models/RefineDet.om";
    initParam.classNum = CLASS_NU;
    initParam.biasesNum = BIASES_NU;
    initParam.biases = "10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326";
    initParam.objectnessThresh = "0.01";
    initParam.iouThresh = "0.45";
    initParam.scoreThresh = "0.1";
    initParam.yoloType = YOLO_TYPE;
    initParam.modelType = 0;
    initParam.inputType = 0;
    initParam.anchorDim = ANCHOR_DIM;
}

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './RefineDetPostProcess test.jpg'.";
        return APP_ERR_OK;
    }

    InitParam initParam;
    InitRefineDetParam(initParam);
    auto RefineDet = std::make_shared<RefineDetDetection>();
    // 初始化模型推理所需的配置信息
    APP_ERROR ret = RefineDet->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "RefineDetDetection init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgPath = argv[1];
    // 推理业务开始
    ret = RefineDet->Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "RefineDetDetection process failed, ret=" << ret << ".";
        RefineDet->DeInit();
        return ret;
    }
    RefineDet->DeInit();
    return APP_ERR_OK;
}