/*
 * Copyright (c) 2022.Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <sys/time.h>
#include <ctime>
#include <stdio.h>
#include <chrono>
#include "MxBase/Log/Log.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include <sys/time.h>
#include <ctime>


namespace {
std::string ReadPipelineConfig(const std::string& pipelineConfigPath)
{
    std::ifstream file(pipelineConfigPath.c_str(), std::ifstream::binary);
    if (!file) {
        LogError << pipelineConfigPath << " file dose not exist.";
        return "";
    }
    file.seekg(0, std::ifstream::end);
    uint32_t fileSize = file.tellg();
    file.seekg(0);
    std::unique_ptr<char[]> data(new char[fileSize]);
    file.read(data.get(), fileSize);
    file.close();
    std::string pipelineConfig(data.get(), fileSize);
    return pipelineConfig;
}
}

int main(int argc, char* argv[])
{
    std::string pipelineConfigPath = "./pipeline/deepsort.pipeline";
    std::string pipelineConfig = ReadPipelineConfig(pipelineConfigPath);
    if (pipelineConfig == "") {
        LogError << "Read pipeline failed.";
        return APP_ERR_COMM_INIT_FAIL;
    }

    MxStream::MxStreamManager mxStreamManager;
    APP_ERROR ret = mxStreamManager.InitManager();
    if (ret != APP_ERR_OK) {
        LogError << "Failed to init Stream manager, ret = " << ret << ".";
        return ret;
    }

    ret = mxStreamManager.CreateMultipleStreams(pipelineConfig);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to create Stream, ret = " << ret << ".";
        return ret;
    }

    FILE *fp = fopen("./out.h264", "wb");
    if (fp == nullptr) {
        LogError << "Failed to open file.";
        return APP_ERR_COMM_OPEN_FAIL;
    }

    bool m_bFoundFirstIDR = false;
    bool bIsIDR = false;
    uint32_t frameCount = 0;
    uint32_t MaxframeCount = 5000;

    std::string streamName = "encoder";
    int inPluginId = 0;
    int msTimeOut = 200000;

    auto start = std::chrono::system_clock::now();
    while (1) {
        MxStream::MxstDataOutput* output = mxStreamManager.GetResult(streamName, inPluginId, msTimeOut);
        if (output == nullptr) {
            LogError << "Failed to get pipeline output.";
            return ret;
        }

         bIsIDR = (output->dataSize > 1);
         if (!m_bFoundFirstIDR) {
             if (!bIsIDR) {
                 continue;
             } else {
                 m_bFoundFirstIDR = true;
             }
         }

        if (fwrite(output->dataPtr, output->dataSize, 1, fp) != 1) {
            LogInfo << "write frame to file fail";
         }
         LogInfo << "Dealing frame id:" << frameCount;
         frameCount++;
         if (frameCount > MaxframeCount) {
             LogInfo << "write frame to file done";
             break;
         }

         delete output;
         auto end = std::chrono::system_clock::now();
         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
         double average = (double)(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den / frameCount;
          std::cout << "fps: " << 1 / average << std::endl;
    }
    
    fclose(fp);
    mxStreamManager.DestroyAllStreams();
    return 0;
}

