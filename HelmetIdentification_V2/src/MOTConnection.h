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

#ifndef MXBASE_HELMETIDENTIFICATION_MOTCONNECTION_H
#define MXBASE_HELMETIDENTIFICATION_MOTCONNECTION_H

#include <thread>
#include <list>
#include <utility>
#include "KalmanTracker.h"
#include "Hungarian.h"
#include "DataType.h"
#include "MxBase/ErrorCode/ErrorCodes.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"
#include "MxBase/postprocess/include/ObjectPostProcessors/Yolov3PostProcess.h"

namespace ascendVehicleTracking {
struct TraceLet {
    TraceInfo info = {};
    int32_t lostAge = 0;
    KalmanTracker kalman;
    std::list<std::pair<DataBuffer, float>> shortFeatureQueue;
    MxBase::ObjectInfo detectInfo = {};
};

class MOTConnection {
public:
    APP_ERROR ProcessSort(std::vector<std::vector<MxBase::ObjectInfo>> &objInfos, size_t frameId);
    APP_ERROR GettrackResult(std::vector<MxBase::ObjectInfo> &objInfos_);

private:
    double trackThreshold_ = 0.3;
    double kIOU_ = 1.0;
    int32_t method_ = 1;
    int32_t lostThreshold_ = 3;
    uint32_t maxNumberFeature_ = 0;
    int32_t generatedId_ = 0;
    std::vector<TraceLet> traceList_ = {};

private:

    void FilterLowThreshold(const HungarianHandle &hungarianHandleObj, const std::vector<std::vector<int>> &disMatrix,
                                std::vector<cv::Point> &matchedTracedDetected, std::vector<bool> &detectVehicleFlagVec);

    void UpdateUnmatchedTraceLet(const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos);

    void UpdateMatchedTraceLet(const std::vector<cv::Point> &matchedTracedDetected,
                                   std::vector<std::vector<MxBase::ObjectInfo>> &objInfos);

    void AddNewDetectedVehicle(std::vector<MxBase::ObjectInfo> &unmatchedVehicleObjectQueue);

    void UpdateTraceLetAndFrame(const std::vector<cv::Point> &matchedTracedDetected,
                                std::vector<std::vector<MxBase::ObjectInfo>> &objInfos, std::vector<MxBase::ObjectInfo> &unmatchedVehicleObjectQueue);

    void TrackObjectPredict();
    void TrackObjectUpdate(const std::vector<std::vector<MxBase::ObjectInfo>> &objInfos,
                           std::vector<cv::Point> &matchedTracedDetected, std::vector<MxBase::ObjectInfo> &unmatchedVehicleObjectQueue);
};
} // namespace ascendVehicleTracking

#endif