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

#include <thread>
#include <pthread.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/cvdef.h>
#include "util.h"

using namespace std;
using namespace cv;

namespace {
const int WIDTH = 1920;
const int HEIGHT = 1080;
const int WIDTHALL = 3840;
const int HEIGHTALL = 2160;
const int VIDEONUM = 4;
const int index0 = 0;
const int index1 = 1;
const int index2 = 2;
const int index3 = 3;
}
// 计算透射变换坐标映射
void getMapping(Mat H, vector<int> &newIndices, vector<int> &orgIndices) {
    int num = HEIGHT * WIDTH;
    int depth = 3;
    Mat old_xy = Mat::ones(num, depth, CV_32F);
    Mat xyz;

    for (int j = 0; j < HEIGHT; j++) {
        for (int i = 0; i < WIDTH; i++) {
            old_xy.at<float>(j * WIDTH + i, index0) = (float)i;
            old_xy.at<float>(j * WIDTH + i, index1) = (float)j;
        }
    }
    Mat H_t, H_32f;
    transpose(H, H_t);
    H_t.convertTo(H_32f, CV_32F);
    xyz = old_xy * H_32f;
    for (int i = 0; i < num; i++) {
        int x = ceil(xyz.at<float>(i, index0) / xyz.at<float>(i, index2));
        int y = ceil(xyz.at<float>(i, index1) / xyz.at<float>(i, index2));
        if (x > 0 & y > 0 & x < WIDTHALL & y < HEIGHTALL) {
            newIndices.emplace_back(y);
            newIndices.emplace_back(x);

            orgIndices.emplace_back(int(old_xy.at<float>(i, index1)));
            orgIndices.emplace_back(int(old_xy.at<float>(i, index0)));
        }
    }
}

// 单张读帧
void ReadFrame(VideoCapture cap, Mat &frame) {
    if (!cap.read(frame)) {
        cout << "读取帧失败" << endl;
        return;
    }    return;
}

void WarpCopy(Mat &src, Mat &warp, int *goalIndices, int *srcIndices, int *goalIndicesEnd) {
    while (goalIndices < goalIndicesEnd) {
        warp.at<Vec3b>(*(goalIndices++), *(goalIndices++)) = src.at<Vec3b>(*(srcIndices++), *(srcIndices++));
    }
}

// 初始化
bool Stitch::Init(vector<String> &videos) {
    for (int i = 0; i < VIDEONUM ;i++) {
        caps_.emplace_back();
        caps_[i].open(videos[i]);
        if (!caps_[i].isOpened())
        {
            cout << "读取视频失败,请检查视频路径！" << endl;
            return false;
        }
        cout << videos[i] << "读取成功！" << endl;
        int width = caps_[i].get(CAP_PROP_FRAME_WIDTH);
        int height = caps_[i].get(CAP_PROP_FRAME_HEIGHT);
        cout << "视频宽度： " << width << endl;
        cout << "视频高度： " << height << endl;
        cout << "视频总帧数： " << caps_[i].get(CAP_PROP_FRAME_COUNT) << endl;
        cout << "帧率： " << caps_[i].get(CAP_PROP_FPS) << endl;
        if (width != WIDTH || height != HEIGHT) {
            cout << "The input videos' resolution must be 1080P, but get " << width << '*' << height << endl;
            return false;
        }
    }
    totalFrames_ = caps_[index0].get(CAP_PROP_FRAME_COUNT);

    if (writeFlag_) {
        double fps = caps_[index0].get(CAP_PROP_FPS);
        writer_ = VideoWriter("./output.avi", VideoWriter::fourcc('x', '2', '6', '4'), fps, Size(WIDTHALL, HEIGHTALL), true);
        if (writer_.isOpened()) {
        cout << "writer_ is opened!" << endl;
        }
        else {
        cout << "writer_ is not opened!" << endl;
        }
    }
    GetTransformationH();
    return true;
}

Stitch::~Stitch() {
    for (int i = 0; i < VIDEONUM ;i++) {
        caps_[i].release();
    }
    writer_.release();
}

// 计算变换矩阵
void CalTransformationH(Mat &img0, Mat &img1, vector<Mat> &Hs) {
    Mat gray0, gray1;
    std::vector<KeyPoint> ipts0, ipts1;
	Mat desp0, desp1;
    cvtColor(img0, gray0, CV_RGB2GRAY);
    clock_t t3 = clock();
    cvtColor(img1, gray1, CV_RGB2GRAY);
    clock_t t4 = clock();
    int minHessian = 2000;    // SURF算法中的hessian阈值
    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(minHessian);
    // 提取特征点并计算特征描述子
	surf->detectAndCompute(gray0, Mat(), ipts0, desp0);
	surf->detectAndCompute(gray1, Mat(), ipts1, desp1);
    // 特征点匹配
    vector<vector<DMatch>> matchPoints;
    FlannBasedMatcher matcher;
    vector<Mat> train_disc(1, desp1);
    matcher.add(train_disc);
    matcher.train();
    matcher.knnMatch(desp0, matchPoints, index2);    // k临近,按顺序排
    vector<DMatch> good_matches;
    for (int i = 0; i < matchPoints.size(); i++) {
        if (matchPoints[i][index0].distance < 0.4f*matchPoints[i][index1].distance)
        {
            good_matches.push_back(matchPoints[i][index0]);
        }
    }
    vector<Point2f> ip0;
    vector<Point2f> ip1;
    // 从匹配成功的匹配对中获取关键点
    for (unsigned int i = 0; i < good_matches.size(); ++i) {
        ip1.push_back(ipts1[good_matches[i].trainIdx].pt);
        ip0.push_back(ipts0[good_matches[i].queryIdx].pt);
    }
    Hs.push_back(findHomography(ip1, ip0, RANSAC));   // 计算透视变换矩阵
}

// 根据首帧计算图像映射关系
bool Stitch::GetTransformationH() {
    Mat warp = Mat(HEIGHTALL, WIDTHALL, CV_8UC3);
    vector<Mat> frame(VIDEONUM);
    vector<Mat> Hs;
    vector<int> newIndices;
    vector<int> orgIndices;

    for (int i = 0; i < VIDEONUM; i++) {
            ReadFrame(caps_[i], frame[i]);
    }

    // 搬运第0路到warp
    frame[index0].copyTo(warp(Rect(0, 0, frame[index0].cols, frame[index0].rows)));

    // 计算第1路坐标映射关系
    CalTransformationH(frame[index0], frame[index1], Hs);
    getMapping(Hs[index0], newIndices, orgIndices);
    goalIndices_[index0] = new int[newIndices.size()];
    memcpy(goalIndices_[index0], &newIndices[index0], newIndices.size() * sizeof(int));
    srcIndices_[index0] = new int[orgIndices.size()];
    memcpy(srcIndices_[index0], &orgIndices[index0], orgIndices.size() * sizeof(int));
    goalIndicesEnd_[index0] = &(goalIndices_[index0][newIndices.size()-1]);

    // 对第1路进行内存搬运
    WarpCopy(frame[index1], warp, goalIndices_[index0], srcIndices_[index0], goalIndicesEnd_[index0]);

    // 计算第2路坐标映射关系
    CalTransformationH(warp, frame[index2], Hs);
    newIndices.clear();
    orgIndices.clear();
    getMapping(Hs[index1], newIndices, orgIndices);
    goalIndices_[index1] = new int[newIndices.size()];
    memcpy(goalIndices_[index1], &newIndices[index0], newIndices.size() * sizeof(int));
    srcIndices_[index1] = new int[orgIndices.size()];
    memcpy(srcIndices_[index1], &orgIndices[index0], orgIndices.size() * sizeof(int));
    goalIndicesEnd_[index1] = &(goalIndices_[index1][newIndices.size()-1]);

    // 对第2路进行内存搬运
    WarpCopy(frame[index2], warp, goalIndices_[index1], srcIndices_[index1], goalIndicesEnd_[index1]);

    // 计算第3路坐标映射关系
    CalTransformationH(warp, frame[index3], Hs);
    newIndices.clear();
    orgIndices.clear();
    getMapping(Hs[index2], newIndices, orgIndices);
    goalIndices_[index2] = new int[newIndices.size()];
    memcpy(goalIndices_[index2], &newIndices[index0], newIndices.size() * sizeof(int));
    srcIndices_[index2] = new int[orgIndices.size()];
    memcpy(srcIndices_[index2], &orgIndices[index0], orgIndices.size() * sizeof(int));
    goalIndicesEnd_[index2] = &(goalIndices_[index2][newIndices.size()-1]);

    // 对第3路进行内存搬运
    WarpCopy(frame[index3], warp, goalIndices_[index2], srcIndices_[index2], goalIndicesEnd_[index2]);

    // 存储拼接后的视频
    if (writeFlag_) {
        writer_.write(warp);
    }
    return true;
}

// 读帧线程函数
void Stitch::ReadWarpped(int frameNums, VideoCapture &cap) {
    for (int i = 0; i < frameNums; i++) {
        Mat warp = Mat(HEIGHTALL, WIDTHALL, CV_8UC3);
        if (!cap.read(warp(Rect(0, 0, WIDTH, HEIGHT)))) {
            cout << "读取帧失败" << endl;
            return;
        }
        {
            unique_lock<mutex> lk(mtx_);
            warps_.push(warp);
            if (warps_.size() == maxSize_) {
                cvRead0_.wait(lk);
            }
        }
        if (warps_.size() == 1) {
            cvStitch0_.notify_one();
        }
    }
}

// 读帧线程函数
void Stitch::Read(int frameNums, VideoCapture &cap, int i) {
    for (int j = 0; j < frameNums; j++) {
        Mat frame;
        ReadFrame(cap, frame);
        switch (i) {
            case index1: {
                {
                    unique_lock<mutex> lk(mtx_);
                    frames1_.push(frame);
                    if (frames1_.size() == maxSize_) {
                        cvRead1_.wait(lk);
                    }
                }
                if (frames1_.size() == 1) {
                    cvStitch1_.notify_one();
                }
                break;
            }
            case index2: {
                {
                    unique_lock<mutex> lk(mtx_);
                    frames2_.push(frame);
                    if (frames2_.size() == maxSize_) {
                        cvRead2_.wait(lk);
                    }
                }
                if (frames2_.size() == 1) {
                    cvStitch2_.notify_one();
                }
                break;
            }
            case index3: {
                {
                    unique_lock<mutex> lk(mtx_);
                    frames3_.push(frame);
                    if (frames3_.size() == maxSize_) {
                        cvRead3_.wait(lk);
                    }
                }
                if (frames3_.size() == 1) {
                    cvStitch3_.notify_one();
                }
                break;
            }
            default :
                cout << "please check input 'i', get " << i << endl;
        }
    }
}

// 拼接线程函数
void Stitch::StitchAll(int frameNums) {
    Mat frame, warp;
    for (int i = 0; i < frameNums; i++) {
        // 第0帧
        {
            unique_lock<mutex> lk(mtx_);
            if (warps_.size() == 0) {
                cvStitch0_.wait(lk);
            }
            warp = warps_.front();
            warps_.pop();
        }
        if (warps_.size() == (maxSize_ - 1)) {
            cvRead0_.notify_one();
        }

        // 第1帧
        {
            unique_lock<mutex> lk(mtx_);
            if (frames1_.size() == 0) {
                cvStitch1_.wait(lk);
            }
            frame = frames1_.front();
            frames1_.pop();
        }
        if (frames1_.size() == (maxSize_ - 1)) {
            cvRead1_.notify_one();
        }
        WarpCopy(frame, warp, goalIndices_[index0], srcIndices_[index0], goalIndicesEnd_[index0]);
        
        // 第2帧
        {
            unique_lock<mutex> lk(mtx_);
            if (frames2_.size() == 0) {
                cvStitch2_.wait(lk);
            }
            frame = frames2_.front();
            frames2_.pop();
        }
        if (frames2_.size() == (maxSize_ - 1)) {
            cvRead2_.notify_one();
        }
        WarpCopy(frame, warp, goalIndices_[index1], srcIndices_[index1], goalIndicesEnd_[index1]);

        // 第3帧
        {
            unique_lock<mutex> lk(mtx_);
            if (frames3_.size() == 0) {
                cvStitch3_.wait(lk);
            }
            frame = frames3_.front();
            frames3_.pop();
        }
        if (frames3_.size() == (maxSize_ - 1)) {
            cvRead3_.notify_one();
        }
        WarpCopy(frame, warp, goalIndices_[index2], srcIndices_[index2], goalIndicesEnd_[index2]);

        // 保存
        if (writeFlag_) {
            writer_.write(warp);
        }
    }
}

// 总调度
bool Stitch::Stitching(int frameNums) {
    thread read0 = thread(&Stitch::ReadWarpped, this, frameNums, ref(caps_[index0]));
    thread read1 = thread(&Stitch::Read, this, frameNums, ref(caps_[index1]), index1);
    thread read2 = thread(&Stitch::Read, this, frameNums, ref(caps_[index2]), index2);
    thread read3 = thread(&Stitch::Read, this, frameNums, ref(caps_[index3]), index3);
    thread stitchAll = thread(&Stitch::StitchAll, this, frameNums);
    if (read0.joinable()) {
        read0.join();
    }
    if (read1.joinable()) {
        read1.join();
    }
    if (read2.joinable()) {
        read2.join();
    }
    if (read3.joinable()) {
        read3.join();
    }
    if (stitchAll.joinable()) {
        stitchAll.join();
    }
    return true;
}