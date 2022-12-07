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

#ifndef UTIL_H
#define UTIL_H
#include <thread>
#include <pthread.h>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <opencv2/opencv.hpp>


class Stitch
{
public:
    explicit Stitch(bool write_video = false, int minHessian = 2000, int maxSize = 2) :
                    writeFlag_(write_video), minHessian_(minHessian), maxSize_(maxSize) {}
    ~Stitch();
    bool Init(std::vector<cv::String> &videos);
    int TotalFrames() {
        return totalFrames_;
    }
    bool Stitching(int frameNums = 0);

private:
    bool GetTransformationH();
    void ReadWarpped(int frameNums, cv::VideoCapture &cap);
    void Read(int frameNums, cv::VideoCapture &cap, int i);
    void StitchAll(int frameNums);

    bool writeFlag_;
    int maxSize_;
    int totalFrames_;
    int minHessian_;
    cv::VideoWriter writer_;
    std::vector<cv::VideoCapture> caps_;
    int *goalIndices_[3];
    int *srcIndices_[3];
    int *goalIndicesEnd_[3];

    std::queue<cv::Mat> warps_;
    std::queue<cv::Mat> frames1_;
    std::queue<cv::Mat> frames2_;
    std::queue<cv::Mat> frames3_;
    std::mutex mtx_;
    std::condition_variable cvRead0_, cvRead1_, cvRead2_, cvRead3_, cvStitch0_, cvStitch1_, cvStitch2_, cvStitch3_;
};

#endif