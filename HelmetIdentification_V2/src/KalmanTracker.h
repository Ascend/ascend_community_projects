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

#ifndef MXBASE_HELMETIDENTIFICATION_KALMANTRACKER_H
#define MXBASE_HELMETIDENTIFICATION_KALMANTRACKER_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include "DataType.h"
#include "MxBase/PostProcessBases/PostProcessDataType.h"

namespace ascendVehicleTracking {
class KalmanTracker {
public:
    KalmanTracker() {}
    ~KalmanTracker() {}
    void CvKalmanInit(MxBase::ObjectInfo initRect);
    MxBase::ObjectInfo Predict();
    void Update(MxBase::ObjectInfo stateMat);
private:
    cv::KalmanFilter cvkalmanfilter_ = {};
    cv::Mat measurement_ = {};
};
} // namesapce ascendVehicleTracking

#endif