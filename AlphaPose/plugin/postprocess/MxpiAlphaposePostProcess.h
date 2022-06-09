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

#ifndef ALPHAPOSEPOSTPROCESS_MXPIALPHAPOSEPOSTPROCESS_H
#define ALPHAPOSEPOSTPROCESS_MXPIALPHAPOSEPOSTPROCESS_H
#include "opencv2/opencv.hpp"
#include "MxBase/ErrorCode/ErrorCode.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxTools/PluginToolkit/base/MxPluginBase.h"
#include "MxTools/PluginToolkit/base/MxPluginGenerator.h"
#include "MxTools/PluginToolkit/metadata/MxpiMetadataManager.h"
#include "../../proto/mxpiAlphaposeProto.pb.h"


/**
* @api
* @brief Definition of MxpiAlphaposePostProcess class.
*/

namespace MxPlugins {
    class MxpiAlphaposePostProcess : public MxTools::MxPluginBase {
    public:
        /**
         * @brief Initialize configure parameter.
         * @param configParamMap
         * @return APP_ERROR
         */
        APP_ERROR Init(std::map<std::string, std::shared_ptr<void> > &configParamMap) override;

        /**
         * @brief DeInitialize configure parameter.
         * @return APP_ERROR
         */
        APP_ERROR DeInit() override;

        /**
         * @brief Process the data of MxpiBuffer.
         * @param mxpiBuffer
         * @return APP_ERROR
         */
        APP_ERROR Process(std::vector<MxTools::MxpiBuffer*> &mxpiBuffer) override;

        /**
         * @brief Definition the parameter of configure properties.
         * @return std::vector<std::shared_ptr<void>>
         */
        static std::vector<std::shared_ptr<void>> DefineProperties();

        /**
         * @brief Overall process to generate all person keypoints information
         * @param srcMxpiObjectList - Source MxpiObjectList containing object data about input image
         * @param srcMxpiTensorPackageList - Source MxpiTensorPackageList containing heatmap data
         * @param dstMxpiPersonList - Target MxpiPersonList containing detection result list
         * @return APP_ERROR
         */
        APP_ERROR GeneratePoseList(const MxTools::MxpiObjectList &srcMxpiObjectList,
                                   const MxTools::MxpiTensorPackageList &srcMxpiTensorPackageList,
                                   mxpialphaposeproto::MxpiPersonList &dstMxpiPersonList);

        /**
         * @brief Prepare output in the format of MxpiPersonList
         * @param finalPoses - Source data containing the information of final keypoints' position
         * @param finalScores - Source data containing the information of fianl keypoints' score
         * @param dstMxpiPersonList - Target data in the format of MxpiPersonList
         * @return APP_ERROR
         */
        APP_ERROR GenerateMxpiOutput(std::vector<cv::Mat> &finalPoses,
                                     std::vector<cv::Mat> &finalScores,
                                     std::vector<float> &personScores,
                                     mxpialphaposeproto::MxpiPersonList &dstMxpiPersonList);
        /**
         * @brief Do maximum suppression to remove redundant keypoints' information
         * @param keypointPreds - Source data containing the information of keypoints position
         * @param keypointScores - Source data containing the information of keypoins score
         * @param objectBoxes - Source data containing the information of object
         * @param finalPoses - Target data containing the information of final keypoints' position
         * @param finalScores - Target data containing the information of fianl keypoints' score
         * @param personScores - Target data containing the information of person's score
         * @return APP_ERROR
         */
        APP_ERROR PoseNms(std::vector<cv::Mat> &keypointPreds,
                          std::vector<cv::Mat> &keypointScores,
                          std::vector<std::vector<float> > &objectBoxes,
                          std::vector<cv::Mat> &finalPoses,
                          std::vector<cv::Mat> &finalScores,
                          std::vector<float> &personScores);
        /**
         * @brief Extract keypoints' location information and scores
         * @param result - Source data containing the information of heatmap data
         * @param objectBoxes - Source data containing the information of object
         * @param keypointPreds - Source data containing the information of keypoints position
         * @param keypointScores - Source data containing the information of keypoins score
         * @return APP_ERROR
         */
        APP_ERROR ExtractKeypointsInfo(const std::vector<std::vector<cv::Mat> > &result,
                                       const std::vector<std::vector<float> > &objectBoxes,
                                       std::vector<cv::Mat> &keypointPreds,
                                       std::vector<cv::Mat> &keypointScores);

    private:
        APP_ERROR SetMxpiErrorInfo(MxTools::MxpiBuffer &buffer, const std::string plugin_name,
                                   const MxTools::MxpiErrorInfo mxpiErrorInfo);
        std::string parentName_;
        std::string objectDetectorName_;
        std::ostringstream ErrorInfo_;
    };
}
#endif // ALPHAPOSEPOSTPROCESS_MXPIALPHAPOSEPOSTPROCESS_H