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

#ifndef MXBASE_HELMETIDENTIFICATION_DATATYPE_H
#define MXBASE_HELMETIDENTIFICATION_DATATYPE_H

#include <memory>
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace ascendVehicleTracking {
#define DVPP_ALIGN_UP(x, align) ((((x) + ((align)-1)) / (align)) * (align))

    const int MODULE_QUEUE_SIZE = 1000;

    enum FrameMode {
        FRAME_MODE_SEARCH = 0,
        FRAME_MODE_REG
    };

    struct DataBuffer {
        std::shared_ptr<uint8_t> deviceData;
        std::shared_ptr<uint8_t> hostData;
        uint32_t dataSize; // buffer size
        DataBuffer() : deviceData(nullptr), hostData(nullptr), dataSize(0) {}
    };

    struct DetectInfo {
        int32_t classId;
        float confidence;
        float minx; // x value of left-top point
        float miny; // y value of left-top point
        float height;
        float width;
    };

    enum TraceFlag {
        NEW_VEHICLE = 0,
        TRACkED_VEHICLE,
        LOST_VEHICLE
    };

    struct TraceInfo {
        int32_t id;
        TraceFlag flag;
        int32_t survivalTime; // How long is it been since the first time, unit: detection period
        int32_t detectedTime; // How long is the vehicle detected, unit: detection period
        std::chrono::time_point<std::chrono::high_resolution_clock> createTime;
    };

    struct TrackLet {
        TraceInfo info;
        // reserved:  kalman status parameter
        int32_t lostTime;                     // undetected time for tracked vehicle
        std::vector<DataBuffer> shortFeature; // nearest 10 frame
    };

    struct VehicleQuality {
        float score;
    };

    struct Coordinate2D {
        uint32_t x;
        uint32_t y;
    };
}
// namespace ascendVehicleTracking

struct AttrT {
    AttrT(std::string name, std::string value) : name(std::move(name)), value(std::move(value)) {}
    std::string name = {};
    std::string value = {};
};

#endif
