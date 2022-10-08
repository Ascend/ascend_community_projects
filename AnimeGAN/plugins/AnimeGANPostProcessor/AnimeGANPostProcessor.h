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

#ifndef ANIMEGANPOSTPROCESSOR_H
#define ANIMEGANPOSTPROCESSOR_H

#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "opencv2/opencv.hpp"

namespace MxPlugins
{
    class AnimeGANPostProcessor : public MxTools::MxPluginBase
    {
    public:
        APP_ERROR Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) override;

        APP_ERROR DeInit() override;

        APP_ERROR Process(std::vector<MxTools::MxpiBuffer *> &mxpiBuffer) override;

        static std::vector<std::shared_ptr<void>> DefineProperties();

    private:
        std::string outputPath_;
    };
}
#endif // ANIMEGANPOSTPROCESSOR_H
