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

#ifndef SDKMEMORY_MXPITRACKIDREPLACECLASSNAME_H
#define SDKMEMORY_MXPITRACKIDREPLACECLASSNAME_H
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"

/**
* @api
* @brief Definition of TrackIdReplaceClassName class.
*/
namespace MxPlugins {
class MxpiTrackIdReplaceClassName : public MxTools::MxPluginBase {
public:
    /**
     * @api
     * @brief Initialize configure parameter.
     * @param configParamMap
     * @return APP_ERROR
     */
    APP_ERROR Init(std::map<std::string, std::shared_ptr<void>>& configParamMap) override;
    /**
     * @api
     * @brief DeInitialize configure parameter.
     * @return APP_ERROR
     */
    APP_ERROR DeInit() override;
    /**
     * @api
     * @brief Process the data of MxpiBuffer.
     * @param mxpiBuffer
     * @return APP_ERROR
     */
    APP_ERROR Process(std::vector<MxTools::MxpiBuffer*>& mxpiBuffer) override;
    /**
     * @api
     * @brief Definition the parameter of configure properties.
     * @return std::vector<std::shared_ptr<void>>
     */
    static std::vector<std::shared_ptr<void>> DefineProperties();
    /**
     * @api
     * @brief Replace className from mxpi_objectpostprocessor with trackId from mxpi_motsimplesortV2
     * @param key
     * @param buffer
     * @return APP_ERROR
     */
    APP_ERROR GenerateSampleOutput(const MxTools::MxpiObjectList srcMxpiObjectList,
                                   const MxTools::MxpiTrackLetList srcMxpiTrackLetList,
                                   MxTools::MxpiObjectList& dstMxpiObjectList);
private:
    APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer& buffer, const std::string pluginName,
    const MxTools::MxpiErrorInfo mxpiErrorInfo);
    APP_ERROR PrintMxpiErrorInfo(MxTools::MxpiBuffer& buffer, const std::string pluginName,
    const MxTools::MxpiErrorInfo mxpiErrorInfo, APP_ERROR app_error, std::string errorName);
    std::string parentName_;
    std::string motName_;
    std::string descriptionMessage_;
    std::ostringstream ErrorInfo_;
};
}
#endif // SDKMEMORY_MXPITRACKIDREPLACECLASSNAME_H
