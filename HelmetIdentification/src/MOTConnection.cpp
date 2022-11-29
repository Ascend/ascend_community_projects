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

#include "MOTConnection.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <map>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "MxBase/Log/Log.h"
#include "MxBase/ErrorCode/ErrorCodes.h"
#include "Hungarian.h"

namespace ascendVehicleTracking
{
    namespace
    {
        // convert double to int
        const int FLOAT_TO_INT = 1000;
        const int MULTIPLE = 0;
        const double SIMILARITY_THRESHOLD = 0.66;
        const int MULTIPLE_IOU = 6;
        const float NORM_EPS = 1e-10;
        const double TIME_COUNTS = 1000.0;
        const double COST_TIME_MS_THRESHOLD = 10.;
        const float WIDTH_RATE_THRESH = 1.f;
        const float HEIGHT_RATE_THRESH = 1.f;
        const float X_DIST_RATE_THRESH = 1.3f;
        const float Y_DIST_RATE_THRESH = 1.f;
    } // namespace

    // 计算bounding box的交并比
    float CalIOU(MxBase::ObjectInfo detect1, MxBase::ObjectInfo detect2)
    {
        cv::Rect_<float> bbox1(detect1.x0, detect1.y0, detect1.x1 - detect1.x0, detect1.y1 - detect1.y0);
        cv::Rect_<float> bbox2(detect2.x0, detect2.y0, detect2.x1 - detect2.x0, detect2.y1 - detect2.y0);
        float intersectionArea = (bbox1 & bbox2).area();
        float unionArea = bbox1.area() + bbox2.area() - intersectionArea;
        if (unionArea < DBL_EPSILON)
        {
            return 0.f;
        }
        return (intersectionArea / unionArea);
    }

    // 计算前后两帧的两个bounding box的相似度
    float CalSimilarity(const TraceLet &traceLet, const MxBase::ObjectInfo &objectInfo, const int &method, const double &kIOU)
    {
        return CalIOU(traceLet.detectInfo, objectInfo);
    }

    // 过滤掉交并比小于阈值的匹配
    void MOTConnection::FilterLowThreshold(const HungarianHandle &hungarianHandleObj,
                                           const std::vector<std::vector<int>> &disMatrix, std::vector<cv::Point> &matchedTracedDetected,
                                           std::vector<bool> &detectVehicleFlagVec)
    {
        for (unsigned int i = 0; i < traceList_.size(); ++i)
        {
            if ((hungarianHandleObj.resX[i] != -1) &&
                (disMatrix[i][hungarianHandleObj.resX[i]] >= (trackThreshold_ * FLOAT_TO_INT)))
            {
                matchedTracedDetected.push_back(cv::Point(i, hungarianHandleObj.resX[i]));
                detectVehicleFlagVec[hungarianHandleObj.resX[i]] = true;
            }
            else
            {
                traceList_[i].info.flag = LOST_VEHICLE;
            }
        }
    }

    // 更新没有匹配上的跟踪器
    void MOTConnection::UpdateUnmatchedTraceLet(const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos)
    {
        for (auto itr = traceList_.begin(); itr != traceList_.end();)
        {
            if ((*itr).info.flag != LOST_VEHICLE)
            {
                ++itr;
                continue;
            }

            (*itr).lostAge++;
            (*itr).kalman.Update((*itr).detectInfo);

            if ((*itr).lostAge < lostThreshold_)
            {
                continue;
            }

            itr = traceList_.erase(itr);
        }
    }

    // 更新匹配上的跟踪器
    void MOTConnection::UpdateMatchedTraceLet(const std::vector<cv::Point> &matchedTracedDetected,
                                              std::vector<std::vector<MxBase::ObjectInfo>> &objInfos)
    {
        for (unsigned int i = 0; i < matchedTracedDetected.size(); ++i)
        {
            int traceIndex = matchedTracedDetected[i].x;
            int detectIndex = matchedTracedDetected[i].y;
            if (traceList_[traceIndex].info.survivalTime > MULTIPLE)
            {
                traceList_[traceIndex].info.flag = TRACkED_VEHICLE;
            }
            traceList_[traceIndex].info.survivalTime++;
            traceList_[traceIndex].info.detectedTime++;
            traceList_[traceIndex].lostAge = 0;
            traceList_[traceIndex].detectInfo = objInfos[0][detectIndex];
            traceList_[traceIndex].kalman.Update(objInfos[0][detectIndex]);
        }
    }
    // 将没有匹配上的检测更新为新的检测器
    void MOTConnection::AddNewDetectedVehicle(std::vector<MxBase::ObjectInfo> &unmatchedVehicleObjectQueue)
    {
        using Time = std::chrono::high_resolution_clock;
        for (auto &vehicleObject : unmatchedVehicleObjectQueue)
        {
            // add new detected into traceList
            TraceLet traceLet;
            generatedId_++;
            traceLet.info.id = generatedId_;
            traceLet.info.survivalTime = 1;
            traceLet.info.detectedTime = 1;
            traceLet.lostAge = 0;
            traceLet.info.flag = NEW_VEHICLE;
            traceLet.detectInfo = vehicleObject;
            traceLet.info.createTime = Time::now();

            traceLet.kalman.CvKalmanInit(vehicleObject);
            traceList_.push_back(traceLet);
        }
    }

    void MOTConnection::UpdateTraceLetAndFrame(const std::vector<cv::Point> &matchedTracedDetected,
                                               std::vector<std::vector<MxBase::ObjectInfo>> &objInfos, std::vector<MxBase::ObjectInfo> &unmatchedVehicleObjectQueue)
    {
        UpdateMatchedTraceLet(matchedTracedDetected, objInfos); // 更新匹配上的跟踪器
        AddNewDetectedVehicle(unmatchedVehicleObjectQueue);     // 将没有匹配上的检测更新为新的检测器
        UpdateUnmatchedTraceLet(objInfos);                      // 更新没有匹配上的跟踪器
    }

    void MOTConnection::TrackObjectPredict()
    {
        // every traceLet should do kalman predict
        for (auto &traceLet : traceList_)
        {
            traceLet.detectInfo = traceLet.kalman.Predict(); // 卡尔曼滤波预测的框
        }
    }

    void MOTConnection::TrackObjectUpdate(const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos,
                                          std::vector<cv::Point> &matchedTracedDetected, std::vector<MxBase::ObjectInfo> &unmatchedVehicleObjectQueue)
    {
        if (objInfos[0].size() > 0)
        {
            LogDebug << "[frame id = " << 1 << "], trace size =" << traceList_.size() << "detect size = " << objInfos[0].size() << "";
            // init vehicle matched flag
            std::vector<bool> detectVehicleFlagVec;
            for (unsigned int i = 0; i < objInfos[0].size(); ++i)
            {
                detectVehicleFlagVec.push_back(false);
            }
            // calculate the associated matrix
            std::vector<std::vector<int>> disMatrix;
            disMatrix.clear();
            disMatrix.resize(traceList_.size(), std::vector<int>(objInfos[0].size(), 0));
            for (unsigned int j = 0; j < objInfos[0].size(); ++j)
            {
                for (unsigned int i = 0; i < traceList_.size(); ++i)
                {
                    // 计算交并比
                    float sim = CalSimilarity(traceList_[i], objInfos[0][j], method_, kIOU_); // method_=1, kIOU_=1.0
                    disMatrix[i][j] = (int)(sim * FLOAT_TO_INT);
                }
            }

            // solve the assignment problem using hungarian  匈牙利算法进行匹配
            HungarianHandle hungarianHandleObj = {};
            HungarianHandleInit(hungarianHandleObj, traceList_.size(), objInfos[0].size());
            HungarianSolve(hungarianHandleObj, disMatrix, traceList_.size(), objInfos[0].size());
            // filter out matched but with low distance  过滤掉匹配上但是交并比较小的
            FilterLowThreshold(hungarianHandleObj, disMatrix, matchedTracedDetected, detectVehicleFlagVec);
            LogDebug << "matchedTracedDetected = " << matchedTracedDetected.size() << "";
            // fill unmatchedVehicleObjectQueue
            for (unsigned int i = 0; i < detectVehicleFlagVec.size(); ++i)
            {
                if (detectVehicleFlagVec[i] == false)
                {
                    unmatchedVehicleObjectQueue.push_back(objInfos[0][i]);
                }
            }
        }
    }

    APP_ERROR MOTConnection::ProcessSort(std::vector<std::vector<MxBase::ObjectInfo>> &objInfos, size_t frameId)
    {
        std::vector<MxBase::ObjectInfo> unmatchedVehicleObjectQueue;
        std::vector<cv::Point> matchedTracedDetected;
        if (objInfos[0].size() == 0)
        {
            return APP_ERR_COMM_FAILURE;
        }

        if (traceList_.size() > 0)
        {
            // every traceLet should do kalman predict
            TrackObjectPredict();                                                            // 卡尔曼滤波预测
            TrackObjectUpdate(objInfos, matchedTracedDetected, unmatchedVehicleObjectQueue); // 选出matched track、unmatched detection
        }
        else
        {
            // traceList is empty, all the vehicle detected in the new frame are unmatched.
            if (objInfos[0].size() > 0)
            {
                for (unsigned int i = 0; i < objInfos[0].size(); ++i)
                {
                    unmatchedVehicleObjectQueue.push_back(objInfos[0][i]);
                }
            }
        }

        // update all the tracelet in the tracelist per frame
        UpdateTraceLetAndFrame(matchedTracedDetected, objInfos, unmatchedVehicleObjectQueue); // 用matched track、unmatched detection更新跟踪器
        return APP_ERR_OK;
    }

    // 获取跟踪后的检测框
    APP_ERROR MOTConnection::GettrackResult(std::vector<MxBase::ObjectInfo> &objInfos_)
    {
        if (traceList_.size() > 0)
        {
            for (auto &traceLet : traceList_)
            {
                traceLet.detectInfo.classId = traceLet.info.id;
                objInfos_.push_back(traceLet.detectInfo);
            }
        }
        else
        {
            return APP_ERR_COMM_FAILURE;
        }
        return APP_ERR_OK;
    }
}
// namespace ascendVehicleTracking