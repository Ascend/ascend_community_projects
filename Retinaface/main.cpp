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
#include <RetinafaceDetection.h>
#include "MxBase/Log/Log.h"

std::string imgPath;
void InitRetinafaceParam(InitParam& initParam)
{
    initParam.deviceId = 0;
    initParam.checkTensor = true;
    initParam.modelPath = "./model/newRetinaface.om";
    initParam.classNum = 1;
    initParam.labelPath = "";
    initParam.ImagePath = imgPath;
}

int main(int argc, char* argv[])
{
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './RetinafacePostProcess test.jpg'.";
        return APP_ERR_OK;
    }
    imgPath = argv[1];

    InitParam initParam;
    InitRetinafaceParam(initParam);
    auto Retinaface = std::make_shared<RetinafaceDetection>();

    APP_ERROR ret = Retinaface->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "RetinafaceDetection init failed, ret=" << ret << ".";
        return ret;
    }

    
    ret = Retinaface->Process(imgPath);
    if (ret != APP_ERR_OK) {
        LogError << "RetinafaceDetection process failed, ret=" << ret << ".";
        Retinaface->DeInit();
        return ret;
    }
    Retinaface->DeInit();
    return APP_ERR_OK;
}