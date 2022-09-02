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

#ifndef SDKMEMORY_PLUGINFEATUREMATCH_H
#define SDKMEMORY_PLUGINFEATUREMATCH_H

#include "opencv2/opencv.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxBase/ErrorCode/ErrorCode.h"


/**
 * This plug is to recognize whether the object's action is a Violent Action and alarm.
*/

namespace MxPlugins {
    class PluginFeatureMatch : public MxTools::MxPluginBase {
    public:
        /**
        * @description: Init configs.
        * @param configParamMap: config.
        * @return: Error code.
        */
        APP_ERROR Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) override;

        /**
        * @description: DeInit device.
        * @return: Error code.
        */
        APP_ERROR DeInit() override;

        /**
        * @description: Plugin_FeatureMatch plugin process.
        * @param mxpiBuffer: data receive from the previous.
        * @return: Error code.
        */
        APP_ERROR Process(std::vector<MxTools::MxpiBuffer *> &mxpiBuffer) override;

        /**
        * @description: Plugin_FeatureMatch plugin define properties.
        * @return: properties.
        */
        static std::vector<std::shared_ptr<void>> DefineProperties();

        /**
        * @api
        * @brief Define the number and data type of input ports.
        * @return MxTools::MxpiPortInfo.
        */
        static MxTools::MxpiPortInfo DefineInputPorts();

        /**
        * @api
        * @brief Define the number and data type of output ports.
        * @return MxTools::MxpiPortInfo.
        */
        static MxTools::MxpiPortInfo DefineOutputPorts();

    private:
        /**
         * @api
         * @brief Check metadata.
         * @param MxTools::MxpiMetadataManager.
         * @return Error Code.
         */
        APP_ERROR CheckDataSource(MxTools::MxpiMetadataManager &mxpiMetadataManager,
                                  MxTools::MxpiMetadataManager &mxpiMetadataManager1);

        APP_ERROR ComputeDistance(MxTools::MxpiTensorPackageList queryTensors,
                                  cv::Mat galleryFeatures, int tensorSize, cv::Mat &distMat);

        void GenerateOutput(MxTools::MxpiObjectList srcObjectList, cv::Mat distMat,
                            int tensorSize, std::vector<std::string> galleryIds,
                            MxTools::MxpiObjectList &dstMxpiObjectList);

        void ReadGalleryFeatures(std::string featuresPath,
                                 std::string idsPath, cv::Mat &galleryFeatures, std::vector<std::string> &galleryIds);
                                
        std::string querySource_ = "";    // previous plugin MxpiClassList
        std::string objectSource_ = "";
        std::string galleryFeaturesPath_ = "";
        std::string galleryIdsPath_ = "";
        std::string metric_ = "";
        std::ostringstream ErrorInfo_;     // Error Code
        float threshold_ = 0.0;
        cv::Mat galleryFeatures = cv::Mat();
        std::vector<std::string> galleryIds = {};
    };
}
#endif