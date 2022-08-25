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

#include "AnimeGANPostProcessor.h"
#include "MxBase/Log/Log.h"
#include <cassert>
#include "time.h"
#include "unistd.h"
using namespace MxPlugins;
using namespace MxTools;
using namespace std;

APP_ERROR AnimeGANPostProcessor::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogInfo << "AnimeGANPostProcessor::Init start.";

    // get `outputPath` parameter from configParamMap
    std::shared_ptr<string> outputPathProSptr =
        std::static_pointer_cast<string>(configParamMap["outputPath"]);
    std::string path = *outputPathProSptr.get();
    outputPath_ = path.back() == '/' ? path : path + "/";

    // create directory if it doesn't exist
    if (access(outputPath_.c_str(), F_OK))
    {
        int ret = system(("mkdir " + outputPath_).c_str());
        if (ret == 0)
        {
            LogInfo << "AnimeGANPostProcessor::Create Output path successfully.";
        }
        else
        {
            LogError << "AnimeGANPostProcessor::Create output path failed.";
            return APP_ERR_FAILURE;
        }
    }

    return APP_ERR_OK;
}

APP_ERROR AnimeGANPostProcessor::DeInit()
{
    LogInfo << "AnimeGANPostProcessor::DeInit end.";

    return APP_ERR_OK;
}

APP_ERROR AnimeGANPostProcessor::Process(std::vector<MxpiBuffer *> &mxpiBuffer)
{
    LogInfo << "AnimeGANPostProcessor::Process start.";

    MxpiBuffer *buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);

    // get metadata and check whether metadata is null
    std::shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(dataSource_);
    if (metadata == nullptr)
    {
        LogError << "AnimeGANPostProcessor: metadata is null.";
        return APP_ERR_METADATA_IS_NULL;
    }

    // check whether the proto struct name is MxpiTensorPackageList
    google::protobuf::Message *msg = (google::protobuf::Message *)metadata.get();
    const google::protobuf::Descriptor *desc = msg->GetDescriptor();
    if (desc->name() != "MxpiTensorPackageList")
    {
        LogError << "AnimeGANPostProcessor: Proto struct name is not MxpiTensorPackageList, failed.";
        return APP_ERR_PROTOBUF_NAME_MISMATCH;
    }

    // process
    shared_ptr<MxpiTensorPackageList> tensorPackageList = static_pointer_cast<MxpiTensorPackageList>(metadata);
    for (int i = 0; i < tensorPackageList->tensorpackagevec_size(); i++)
    {
        MxpiTensorPackage tensorPackage = tensorPackageList->tensorpackagevec(i);

        for (int j = 0; j < tensorPackage.tensorvec_size(); j++)
        {
            MxpiTensor tensor = tensorPackage.tensorvec(j);
            vector<int> shape;

            for (int k = 0; k < tensor.tensorshape_size(); k++)
            {
                shape.push_back((int)tensor.tensorshape(k));
            }

            // set shape for converting
            auto b = shape[0];
            auto h = shape[1];
            auto w = shape[2];
            auto c = shape[3];

            // convert data from MxpiTensor to openCV's Mat
            cv::Mat img = cv::Mat(cv::Size(w, h), CV_32FC3, ((float32_t *)tensor.tensordataptr()));
            cv::Mat temp, output;

            // convert data from [-1,1] to [0,255] and dtype from float32 to uint8
            int alpha = 127.5;
            int beta = 127.5;
            img.convertTo(temp, CV_8UC3, alpha, beta);

            // convert color from BGR to RGB
            cv::cvtColor(temp, output, cv::COLOR_BGR2RGB);

            // save results to outputPath_,filename is decided by current time
            time_t t;
            time(&t);
            string savePath = outputPath_.back() == '/' ? outputPath_ + "output_" + to_string(t) + ".jpg" : outputPath_ + "/output_" + to_string(t) + ".jpg";
            cv::imwrite(savePath, output);
            LogInfo << "AnimeGANPostProcessor::Output Image is saved in:" << savePath;
        }
    }
    SendData(0, *buffer);

    LogInfo << "AnimeGANPostProcessor::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> AnimeGANPostProcessor::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;

    // register new property `outputPath`
    auto outputPathProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string>
    {STRING, "outputPath", "path", "Where the output image should be saved", "./output.jpg", "NULL", "NULL"});
    properties.push_back(outputPathProSptr);

    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(AnimeGANPostProcessor)
