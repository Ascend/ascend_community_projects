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

#include <unistd.h>
#include <thread>
#include <pthread.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "util.h"

using namespace std;
using namespace cv;

void MainStitch(vector<String> &videos, int frameNums, bool writeVideo, int minHessian) {
    // 初始化
    Stitch stitch(writeVideo, minHessian = minHessian);
    bool ret = stitch.Init(videos);
    if (!ret) {
        cout << "初始化失败！";
        return;
    }
    if ((frameNums < 1) | (frameNums >= stitch.TotalFrames())) {
        frameNums = stitch.TotalFrames() - 1;
    }
    cout << "初始化成功！" << endl;
    // 拼接
    clock_t sumBegin = clock();
    stitch.Stitching(frameNums);
    clock_t sumFinish = clock();
    if (!writeVideo) {
        cout << "端到端平均耗时：" <<  float(sumFinish - sumBegin) / CLOCKS_PER_SEC / frameNums << " seconds" << endl;
    }
}

int main(int argc, char* argv[]) {
    int videoNum = 4;
    vector<String> videos(videoNum);
    int maxSize = 2;
    int frames = 0;
    bool writeVideo = false;
    int argNum = 8;
    int minHessian;
    int minHessianLimit = 10000;
    if (argc < argNum) {
        cout << "missing parameter!" << endl;
        return 0;
    }
    int argIndex = 1;
    if (atoi(argv[argIndex])) {
        frames = atoi(argv[argIndex]);
    }
    if (atoi(argv[++argIndex])) {
        writeVideo = true;
    }
    if (atoi(argv[++argIndex])) {
        minHessian = atoi(argv[argIndex]);
    }
    int videoIndex = 0;
    videos[videoIndex++] = argv[++argIndex];
    videos[videoIndex++] = argv[++argIndex];
    videos[videoIndex++] = argv[++argIndex];
    videos[videoIndex++] = argv[++argIndex];
    
    if (minHessian <= 0 || minHessian >= minHessianLimit) {
        cout << "minHessian must be in range (0,10000), but get " << minHessian << endl;
        return -1;
    }
    for (int i = 0; i < videoNum; i++) {
        if (access(videos[i].c_str(), F_OK) == -1) {
            cout << "File " << videos[i] << " does not exist.\n";
            return -1;
        }
        string suffix_str = videos[i].substr(videos[i].find_last_of('.') + 1);
        if (suffix_str != "mp4" & suffix_str != "MP4") {
            cout << "File " << videos[i] << " isn't MP4.\n";
            return -1;
        }
    }
    MainStitch(videos, frames, writeVideo, minHessian);
    return 0;
}