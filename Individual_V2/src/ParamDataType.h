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

#ifndef PARAMDATATYPE
#define PARAMDATATYPE
#include "MxBase/MxBase.h"

struct V2Param
{
    uint32_t deviceId;
    std::string labelPath;
    std::string configPath;
    std::string modelPath;
    V2Param() {}
    V2Param(const uint32_t &deviceId, const std::string &labelPath, const std::string &configPath, const std::string &modelPath)
        : deviceId(deviceId), labelPath(labelPath), configPath(configPath), modelPath(modelPath) {}
};
#endif