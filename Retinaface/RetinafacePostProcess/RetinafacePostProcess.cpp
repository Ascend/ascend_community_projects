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

#include "RetinafacePostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include <map>

namespace {
    const uint32_t LEFTTOPX = 0;
    const uint32_t LEFTTOPY = 1;
    const uint32_t RIGHTTOPX = 2;
    const uint32_t RIGHTTOPY = 3;
    const int PRIOR_PARAMETERS[3][2] = {{16, 32}, {64, 128}, {256, 512}};
    const int PRIOR_PARAMETERS_COUNT = 2;
    const float IMAGE_WIDTH = 1000.0;
    const float IMAGE_HEIGHT = 1000.0;
    const float STEPS[3] = {8.0, 16.0, 32.0};
    const float VARIANCE[2] = {0.1, 0.2};
    const uint32_t RECTANGLEPOINT = 4;
    const uint32_t KEYPOINTNUM = 5;
    const uint32_t POINT_SIZE = 1;
    const uint32_t DIM = 2;
    const uint32_t RECTANGLE_COLOR = 1;
    const uint32_t KEYPOINT_COLOR = 2;
    const uint32_t DIV_TWO = 2;
    uint32_t ORIGIALWIDE;
    uint32_t ORIGINALHEIGHT;
}
using namespace MxBase;

RetinafacePostProcess& RetinafacePostProcess::operator=(const RetinafacePostProcess& other) {
    if (this == &other) {
        return *this;
    }
    ObjectPostProcessBase::operator=(other);
    return *this;
}

APP_ERROR RetinafacePostProcess::Init(const std::map<std::string, std::shared_ptr<void>>& postConfig, const int& OriginWide, const int& OriginHeight) {
    LogDebug << "Start to Init RetinafacePostProcess.";
    ORIGIALWIDE = OriginWide;
    ORIGINALHEIGHT = OriginHeight;
    APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
        return ret;
    }
    LogInfo << "End to Init RetinafacePostprocess.";
    return APP_ERR_OK;
}

APP_ERROR RetinafacePostProcess::DeInit() {
    return APP_ERR_OK;
}

void RetinafacePostProcess::ObjectDetectionOutput(const std::vector <TensorBase>& tensors,
    std::vector <std::vector<ObjectInfo>>& objectInfos,
    const std::vector <ResizedImageInfo>& resizedImageInfos)
{
    LogInfo << "RetinafacePostProcess start to write results.";
    std::cout << "\n";
    std::cout << "tensorsSize =" << tensors.size() << "\n";
    for (uint32_t i = 0; i<tensors.size(); i++) {
        for (uint32_t j = 0; j<tensors[i].GetShape().size(); j++) {
            std::printf("tensors[%d][%d] = %d .", i, j, tensors[i].GetShape()[j]);
        }
        std::cout << std::endl;
    }
    for (auto num : { objectInfoTensor_, objectConfTensor_ }) {
            if ((num >= tensors.size()) || (num < 0)) {
                LogError << GetError(APP_ERR_INVALID_PARAM) << "TENSOR(" << num
                    << ") must ben less than tensors'size(" << tensors.size() << ") and larger than 0.";
            }
        }
        auto shape = tensors[0].GetShape();
        auto keyshape = tensors[2].GetShape();

        cv::Mat PriorBox;
        cv::Mat location = cv::Mat(shape[1], shape[2], CV_32FC1, tensors[0].GetBuffer());
        cv::Mat keylocation = cv::Mat(keyshape[1], keyshape[2], CV_32FC1, tensors[2].GetBuffer());
        GeneratePriorBox(PriorBox);

        float width_resize = 1000;
        float height_resize = 1000;
        float width_original = 1000;
        float height_original = 1000;
        float width_resize_scale = width_resize / (width_original * 1.0);
        float height_resize_scale = height_resize / (height_original * 1.0);
        float resize_scale_factor = 1.0;
        if (width_resize_scale >= height_resize_scale) {
            resize_scale_factor = height_resize_scale;
        } else {
            resize_scale_factor = width_resize_scale;
        }

        cv::Mat res = decode_for_loc(location, PriorBox, keylocation, resize_scale_factor);

        uint32_t batchSize = shape[0];
        uint32_t VectorNum = shape[1];
		
        std::map<ObjectInfo, KeyPointDetectionInfo> match;
        for (uint32_t i = 0; i < batchSize; i++) {
            std::vector <ObjectInfo> objectInfo;
            std::vector <ObjectInfo> objectInfoSorted;
            std::vector <KeyPointDetectionInfo> keypointInfo;
            std::vector <KeyPointDetectionInfo> keypointInfoSorted;
            auto dataPtr_Conf = (float *) tensors[1].GetBuffer() + i * tensors[1].GetByteSize() / batchSize;

            for (uint32_t j = 0; j < VectorNum; j++) {
                float* begin_Conf = dataPtr_Conf + j * 2;
                float conf = *(begin_Conf + 1);

                if (conf > confThresh_) {
                    ObjectInfo objInfo;
                    objInfo.confidence = j;
                    objInfo.x0 = res.at<float>(j, LEFTTOPX) * IMAGE_WIDTH / width_resize_scale;
                    objInfo.y0 = res.at<float>(j, LEFTTOPY) * IMAGE_HEIGHT / height_resize_scale;
                    objInfo.x1 = res.at<float>(j, RIGHTTOPX) * IMAGE_WIDTH / width_resize_scale;
                    objInfo.y1 = res.at<float>(j, RIGHTTOPY) * IMAGE_HEIGHT / height_resize_scale;
                    objInfo.classId = RECTANGLE_COLOR;

                    objectInfo.push_back(objInfo);
                }
            }
            MxBase::NmsSort(objectInfo, iouThresh_);

            for (uint32_t j = 0; j < objectInfo.size(); j++) {
            	ObjectInfo obj = objectInfo[j];
                KeyPointDetectionInfo kpInfo;
                int keypoint_Pos = objectInfo[j].confidence;
                float* begin_Conf = dataPtr_Conf + keypoint_Pos * 2;
                float conf = *(begin_Conf + 1);
                objectInfo[j].confidence = conf;
                objectInfoSorted.push_back(objectInfo[j]);

                for (uint32_t k = 0; k < KEYPOINTNUM; k++)
                {
                    float x = res.at<float>(keypoint_Pos, RECTANGLEPOINT + k * DIM) * IMAGE_WIDTH / width_resize_scale;
                    float y = res.at<float>(keypoint_Pos, RECTANGLEPOINT + k * DIM + 1) * IMAGE_HEIGHT / height_resize_scale;
                    ObjectInfo objInfo;

                    objInfo.x0= x - POINT_SIZE;
                    objInfo.x1= x + POINT_SIZE;
                    objInfo.y0= y - POINT_SIZE;
                    objInfo.y1= y + POINT_SIZE;
                    objInfo.confidence = 0;
                    objInfo.classId = KEYPOINT_COLOR;
                    objectInfoSorted.push_back(objInfo);
                }
            }

            objectInfos.push_back(objectInfoSorted);
        }
        LogInfo << "Retinaface write results successed.";
}
APP_ERROR RetinafacePostProcess::Process(const std::vector<TensorBase>& tensors,
    std::vector<std::vector<ObjectInfo>>& objectInfos,
    const std::vector<ResizedImageInfo>& resizedImageInfos,
    const std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "Start to Process RetinafacePostProcess.";
    APP_ERROR ret = APP_ERR_OK;
    auto inputs = tensors;
    ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed. ret=" << ret;
        return ret;
    }
    ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);
    LogInfo << "End to Process RetinafacePostProcess.";
    return APP_ERR_OK;
}

void RetinafacePostProcess::GeneratePriorBox(cv::Mat& anchors)
{
    std::vector<std::vector<int>>feature_maps(RIGHTTOPY, std::vector<int>(DIM));
    for (uint32_t i = 0; i < feature_maps.size(); i++) {
        feature_maps[i][0] = ceil(IMAGE_HEIGHT / STEPS[i]);
        feature_maps[i][1] = ceil(IMAGE_WIDTH / STEPS[i]);
    }
    for (uint32_t k = 0; k < feature_maps.size(); k++) {
        auto f = feature_maps[k];

        float step = (float)STEPS[k];
        for (int i = 0; i < f[0]; i++) {
            for (int j = 0; j < f[1]; j++) {
                for (int l = 0; l < PRIOR_PARAMETERS_COUNT && PRIOR_PARAMETERS[k][l] != -1; l++) {
                    float min_size = PRIOR_PARAMETERS[k][l];
                    cv::Mat anchor(1, RECTANGLEPOINT * DIM, CV_32F);
                    float center_x = (j + 0.5f) * step;
                    float center_y = (i + 0.5f) * step;

                    float xmin = (center_x - min_size / 2.f) / IMAGE_WIDTH;
                    float ymin = (center_y - min_size / 2.f) / IMAGE_HEIGHT;
                    float xmax = (center_x + min_size / 2.f) / IMAGE_WIDTH;
                    float ymax = (center_y + min_size / 2.f) / IMAGE_HEIGHT;

                    float prior_width = xmax - xmin;
                    float prior_height = ymax - ymin;
                    float prior_center_x = (xmin + xmax) / 2;
                    float prior_center_y = (ymin + ymax) / 2;

                    anchor.at<float>(0, LEFTTOPX) = center_x / IMAGE_WIDTH;
                    anchor.at<float>(0, LEFTTOPY) = center_y / IMAGE_HEIGHT;
                    anchor.at<float>(0, RIGHTTOPX) = min_size / IMAGE_WIDTH;
                    anchor.at<float>(0, RIGHTTOPY) = min_size / IMAGE_HEIGHT;

                    anchor.at<float>(0, LEFTTOPX + RECTANGLEPOINT) = prior_width;
                    anchor.at<float>(0, LEFTTOPY + RECTANGLEPOINT) = prior_height;
                    anchor.at<float>(0, RIGHTTOPX + RECTANGLEPOINT) = prior_center_x;
                    anchor.at<float>(0, RIGHTTOPY + RECTANGLEPOINT) = prior_center_y;

                    anchors.push_back(anchor);
                }
            }
        }
    }
}

cv::Mat RetinafacePostProcess::decode_for_loc(cv::Mat& loc, cv::Mat& prior, cv::Mat& key, float resize_scale_factor) {
    LogInfo << loc.rows;
    LogInfo << loc.cols;
    LogInfo << prior.rows;
    LogInfo << prior.cols;
    LogInfo << key.rows;
    LogInfo << key.cols;
    cv::Mat loc_first = loc.colRange(0, 2);
    cv::Mat loc_last = loc.colRange(2, 4);
    cv::Mat prior_first = prior.colRange(0, 2);
    cv::Mat prior_last = prior.colRange(2, 4);
    cv::Mat prior_first2 = prior.colRange(4, 6);
    cv::Mat prior_last2 = prior.colRange(6, 8);
    cv::Mat facepoint = key.colRange(0, 10);
    cv::Mat boxes1 = prior_first + (loc_first * VARIANCE[0]).mul(prior_last);
    cv::Mat boxes2;
    cv::exp(loc_last * VARIANCE[1], boxes2);
    boxes2 = boxes2.mul(prior_last);
    boxes1 = boxes1 - boxes2 / DIV_TWO;
    boxes2 = boxes2 + boxes1;
    cv::Mat boxes3;
    for (uint32_t i = 0; i < KEYPOINTNUM; i++)
    {
        cv::Mat singlepoint = facepoint.colRange(i * 2, (i + 1) * 2);
        singlepoint = prior_last2 + (singlepoint * VARIANCE[0]).mul(prior_first2);
        if (i == 0) boxes3 = singlepoint;
        else cv::hconcat(boxes3, singlepoint, boxes3);
    }

    cv::Mat boxes;
    cv::hconcat(boxes1, boxes2, boxes);
    cv::hconcat(boxes, boxes3, boxes);
    if (resize_scale_factor == 0) {
        LogError << "resize_scale_factor is 0.";
    }
    return boxes;
}