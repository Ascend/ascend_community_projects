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

#include "fstream"
#include "AttrRecognition.h"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/PostProcessBases/ClassPostProcessBase.h"

AttrRecognition::AttrRecognition(const V2Param &v2Param)
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

    std::ifstream readConfig;
    std::ifstream readLabels;
    std::string line;

    // read config file
    readConfig.open(configPath, std::ios::in);
    while (getline(readConfig, line))
    {
        if (boost::starts_with(line, "SOFTMAX="))
        {
            std::vector<std::string> words;
            auto value = boost::split(words, line, boost::is_any_of("=")).back();

            value = boost::to_lower_copy(value);
            if (value == "true")
            {
                softmaxFlag = true;
            }
            else if (value == "false")
            {
                softmaxFlag = false;
            }
            else
            {
                LogError << "Invalid value of SOFTMAX in config file.Use default value `false`.";
            }
        }
        else if (boost::starts_with(line, "CLASS_NUM="))
        {
            std::vector<std::string> words;

            auto value = boost::split(words, line, boost::is_any_of("=")).back();
            if (!boost::all(value, boost::is_digit()))
            {
                LogError << "Invalid value of SOFTMAX in config file.Use default value `40`.";
            }
            else
            {
                classNum = std::stoi(value);
            }
        }
        else if (boost::starts_with(line, "TOPK="))
        {
            std::vector<std::string> words;
            
            auto value = boost::split(words, line, boost::is_any_of("=")).back();
            if (!boost::all(value, boost::is_digit()))
            {
                LogError << "Invalid value of TOPK in config file.Use default value `40`.Please check it.";
            }
            else
            {
                topK = std::stoi(value);
            }
        }
    }
    readConfig.close();

    // read label file
    uint32_t nums = 0;
    readLabels.open(labelPath, std::ios::in);
    while (getline(readLabels, line))
    {
        if (!boost::starts_with(line, "#"))
        {
            labels.push_back(line);
            nums++;
            if (nums >= classNum)
            {
                break;
            }
        }
    }
    readLabels.close();
};

APP_ERROR AttrRecognition::Infer(MxBase::Image &resizeImage, std::vector<MxBase::Tensor> &outputs)
{
    APP_ERROR ret;

    // !move image to device!
    MxBase::Tensor tensorImg = resizeImage.ConvertToTensor();
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

APP_ERROR AttrRecognition::PostProcess(std::vector<MxBase::Tensor> &outputs, std::vector<std::vector<MxBase::ClassInfo>> &classInfos)
{
    std::vector<MxBase::TensorBase> tensors;
    for (size_t i = 0; i < outputs.size(); i++)
    {
        MxBase::MemoryData memoryData(outputs[i].GetData(), outputs[i].GetByteSize());
        MxBase::TensorBase tensorBase(memoryData, true, outputs[i].GetShape(), MxBase::TENSOR_DTYPE_INT32);
        tensors.push_back(tensorBase);
    }

    auto inputs = outputs;

    const uint32_t softmaxTensorIndex = 0;
    auto softmaxTensor = inputs[softmaxTensorIndex];
    auto shape = softmaxTensor.GetShape();
    uint32_t batchSize = shape[0];
    void *softmaxTensorPtr = softmaxTensor.GetData();
    uint32_t topk = std::min(topK, classNum);

    for (uint32_t i = 0; i < batchSize; i++)
    {
        std::vector<uint32_t> idx = {};
        for (uint32_t j = 0; j < classNum; j++)
        {
            idx.push_back(j);
        }

        std::vector<float> softmax = {};
        for (uint32_t j = 0; j < classNum; j++)
        {
            float value = *((float *)softmaxTensorPtr + i * classNum + j);
            softmax.push_back(value);
        }
        if (softmaxFlag)
        {
            fastmath::softmax(softmax);
        }

        auto cmp = [&softmax](uint32_t index_1, uint32_t index_2)
        {
            return softmax[index_1] > softmax[index_2];
        };

        std::sort(idx.begin(), idx.end(), cmp);

        std::vector<MxBase::ClassInfo> topkClassInfos = {};
        double ATTR_CONFIDENCE = 0.5;
        for (uint32_t j = 0; j < topk; j++)
        {
            MxBase::ClassInfo clsInfo = {};
            if (softmax[j] > ATTR_CONFIDENCE)
            {
                clsInfo.classId = j;
                clsInfo.confidence = 1;
                clsInfo.className = labels[j];
            }
            else
            {
                clsInfo.classId = j;
                clsInfo.confidence = 0;
                clsInfo.className = labels[j];
            }
            topkClassInfos.push_back(clsInfo);
        }
        classInfos.push_back(topkClassInfos);
    }
    return APP_ERR_OK;
};
