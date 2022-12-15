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

#include "KalmanTracker.h"

namespace ascendVehicleTracking
{
    namespace
    {
        const int OFFSET = 2;
        const int MULTIPLE = 2;
    }

    /*
     * The SORT algorithm uses a linear constant velocity model,which assumes 7
     * states, including
     * x coordinate of bounding box center
     * y coordinate of bounding box center
     * area of bounding box
     * aspect ratio of w to h
     * velocity of x
     * velocity of y
     * variation rate of area
     *
     * The aspect ratio is considered to be unchanged, so there is no additive item
     * for aspect ratio in the transitionMatrix
     *
     *
     * Kalman filter equation step by step
     * (1)  X(k|k-1)=AX(k-1|k-1)+BU(k)
     * X(k|k-1) is the predicted state(statePre),X(k-1|k-1) is the k-1 statePost,A
     * is transitionMatrix, B is controlMatrix, U(k) is control state, in SORT U(k) is 0.
     *
     * (2)  P(k|k-1)=AP(k-1|k-1)A'+Q
     * P(k|k-1) is the predicted errorCovPre, P(k-1|k-1) is the k-1 errorCovPost,
     * Q is processNoiseCov
     *
     * (3)  Kg(k)=P(k|k-1)H'/(HP(k|k-1))H'+R
     * Kg(k) is the kalman gain, the ratio of estimate variance in total variance,
     * H is the measurementMatrix,R is the measurementNoiseCov
     *
     * (4)  X(k|k)=X(k|k-1)+Kg(k)(Z(k)-HX(k|k-1))
     * X(k|k) is the k statePost, Z(k) is the measurement of K, in SORT Z(k) is
     * the detection result of k
     *
     * (5)  P(k|k)=(1-Kg(k)H)P(k|k-1)
     * P(k|k) is the errorCovPost
     */
    void KalmanTracker::CvKalmanInit(MxBase::ObjectInfo initRect)
    {
        const int stateDim = 7;
        const int measureDim = 4;
        cvkalmanfilter_ = cv::KalmanFilter(stateDim, measureDim, 0); // zero control
        measurement_ = cv::Mat::zeros(measureDim, 1, CV_32F);        // 4 measurements, Z(k), according to detection results
        // A, will not be updated
        cvkalmanfilter_.transitionMatrix = (cv::Mat_<float>(stateDim, stateDim) << 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1);
        cvkalmanfilter_.measurementMatrix = (cv::Mat_<float>(measureDim, stateDim) << 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0);
        cv::setIdentity(cvkalmanfilter_.measurementMatrix);                          // H, will not be updated
        cv::setIdentity(cvkalmanfilter_.processNoiseCov, cv::Scalar::all(1e-2));     // Q, will not be updated
        cv::setIdentity(cvkalmanfilter_.measurementNoiseCov, cv::Scalar::all(1e-1)); // R, will bot be updated
        cv::setIdentity(cvkalmanfilter_.errorCovPost, cv::Scalar::all(1));           // P(k-1|k-1), will be updated

        // initialize state vector with bounding box in
        // [center_x,center_y,area,ratio]
        // style, the velocity is 0
        // X(k-1|k-1)
        cvkalmanfilter_.statePost.at<float>(0, 0) = initRect.x0 + (initRect.x1 - initRect.x0) / MULTIPLE;
        cvkalmanfilter_.statePost.at<float>(1, 0) = initRect.y0 + (initRect.y1 - initRect.y0) / MULTIPLE;
        cvkalmanfilter_.statePost.at<float>(OFFSET, 0) = (initRect.x1 - initRect.x0) * (initRect.y1 - initRect.y0);
        cvkalmanfilter_.statePost.at<float>(OFFSET + 1, 0) = (initRect.x1 - initRect.x0) / (initRect.y1 - initRect.y0);
    }

    // Predict the bounding box.
    MxBase::ObjectInfo KalmanTracker::Predict()
    {
        // predict
        // return X(k|k-1)=AX(k-1|k-1), and update
        // P(k|k-1) <- AP(k-1|k-1)A'+Q
        MxBase::ObjectInfo detectInfo = {};
        cv::Mat predictState = cvkalmanfilter_.predict();
        float *pData = (float *)(predictState.data);
        float w = sqrt((*(pData + OFFSET)) * (*(pData + OFFSET + 1)));
        if (w < DBL_EPSILON)
        {
            detectInfo.x0 = 0;
            detectInfo.y0 = 0;
            detectInfo.x1 = 0;
            detectInfo.y1 = 0;
            detectInfo.classId = 0;
            return detectInfo;
        }
        if (w == 0)
        {
            MxBase::ObjectInfo w0DetectInfo = {};
            return w0DetectInfo;
        }
        float h = (*(pData + OFFSET)) / w;
        float x = (*pData) - w / MULTIPLE;
        float y = (*(pData + 1)) - h / MULTIPLE;
        if (x < 0 && (*pData) > 0)
        {
            x = 0;
        }
        if (y < 0 && (*(pData + 1)) > 0)
        {
            y = 0;
        }
        detectInfo.x0 = x;
        detectInfo.y0 = y;
        detectInfo.x1 = x + w;
        detectInfo.y1 = y + h;
        return detectInfo;
    }

    // Update the state using observed bounding box
    void KalmanTracker::Update(MxBase::ObjectInfo stateMat)
    {
        // measurement_, update Z(k)
        float *pData = (float *)(measurement_.data);
        *pData = stateMat.x0 + (stateMat.x1 - stateMat.x0) / MULTIPLE;
        *(pData + 1) = stateMat.y0 + (stateMat.y1 - stateMat.y0) / MULTIPLE;
        *(pData + OFFSET) = (stateMat.x1 - stateMat.x0) * (stateMat.y1 - stateMat.y0);
        *(pData + OFFSET + 1) = (stateMat.x1 - stateMat.x0) / (stateMat.y1 - stateMat.y0);
        // update, do the following steps:
        // Kg(k): P(k|k-1)H'/(HP(k|k-1))H'+R
        // X(k|k): X(k|k-1)+Kg(k)(Z(k)-HX(k|k-1))
        // P(k|k): (1-Kg(k)H)P(k|k-1)
        cvkalmanfilter_.correct(measurement_);
    }
} // namespace
