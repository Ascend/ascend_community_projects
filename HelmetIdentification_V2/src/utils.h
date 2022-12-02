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

#ifndef MXBASE_HELMETIDENTIFICATION_UTILS_H
#define MXBASE_HELMETIDENTIFICATION_UTILS_H

#include <iostream>
#include <map>
#include <fstream>
#include "unistd.h"
#include <memory>
#include <queue>
#include <thread>
#include "boost/filesystem.hpp"

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/MemoryHelper/MemoryHelper.h"
#include "opencv2/opencv.hpp"
#include "MxBase/postprocess/include/ObjectPostProcessors/Yolov3PostProcess.h"

#include "MxBase/E2eInfer/ImageProcessor/ImageProcessor.h"
#include "MxBase/E2eInfer/VideoDecoder/VideoDecoder.h"
#include "MxBase/E2eInfer/VideoEncoder/VideoEncoder.h"
#include "MxBase/E2eInfer/DataType.h"

#include "MxBase/MxBase.h"
#include "MOTConnection.h"
#include "cropResizePaste.hpp"
#include "chrono"
#include "time.h"

struct FrameImage
{
    MxBase::Image image;
    uint32_t frameId = 0;
    uint32_t channelId = 0;
};

namespace videoInfo
{
    const uint32_t SRC_WIDTH = 1920;
    const uint32_t SRC_HEIGHT = 1080;

    const uint32_t YUV_BYTE_NU = 3;
    const uint32_t YUV_BYTE_DE = 2;
    const uint32_t YOLOV5_RESIZE = 640;

    // 要检测的目标类别的标签
    const std::string TARGET_CLASS_NAME = "head";
    // 使用chrono计数结果为毫秒，需要除以1000转换为秒
    double MS_PPE_SECOND = 1000.0;

    const uint32_t DEVICE_ID = 0;
    std::string labelPath = "/home/HwHiAiUser/testmain/HelmetIdentification_V2/model/imgclass.names";
    std::string configPath = "/home/HwHiAiUser/testmain/HelmetIdentification_V2/model/Helmet_yolov5.cfg";
    std::string modelPath = "/home/HwHiAiUser/testmain/HelmetIdentification_V2/model/YOLOv5_s.om";

    // 读入视频帧Image的队列
    std::queue<FrameImage> frameImageQueue;
    // 读入视频帧Image队列的线程锁
    std::mutex g_threadsMutex_frameImageQueue;

    // resize前即原始图像的vector
    std::vector<FrameImage> realImageVector;
    // resize后Image的vector
    std::vector<FrameImage> resizedImageVector;
    // resize后Image队列的线程锁
    std::mutex g_threadsMutex_resizedImageVector;

    // 推理后tensor的vector
    std::vector<std::vector<MxBase::Tensor>> inferOutputVector;
    // 推理后tensor队列的线程锁
    std::mutex g_threadsMutex_inferOutputVector;

    // 后处理后objectInfos的队列 map<channelId, map<frameId, ObjectInfo的vector>>
    std::vector<std::vector<std::vector<MxBase::ObjectInfo>>> postprocessOutputVector;
    // 后处理后objectInfos队列的线程锁
    std::mutex g_threadsMutex_postprocessOutputVector;
}
namespace fs = boost::filesystem;

// resize线程
void resizeMethod(std::chrono::high_resolution_clock::time_point start_time, size_t target_count)
{
    std::chrono::high_resolution_clock::time_point resize_start_time = std::chrono::high_resolution_clock::now();

    MxBase::ImageProcessor imageProcessor(videoInfo::DEVICE_ID);

    size_t resize_count = 0;
    while (resize_count < target_count)
    {
        if (videoInfo::frameImageQueue.empty())
        {
            continue;
        }
        // 取图像并resize
        FrameImage frame = videoInfo::frameImageQueue.front();
        MxBase::Image image = frame.image;
        MxBase::Size originalSize = image.GetOriginalSize();
        MxBase::Size resizeSize = MxBase::Size(videoInfo::YOLOV5_RESIZE, videoInfo::YOLOV5_RESIZE);
        MxBase::Image outputImage = resizeKeepAspectRatioFit(originalSize.width, originalSize.height, resizeSize.width, resizeSize.height, image, imageProcessor);
        // 先将缩放后的图像放入resizeImage的队列
        FrameImage resizedFrame;
        resizedFrame.channelId = frame.channelId;
        resizedFrame.frameId = frame.frameId;
        resizedFrame.image = outputImage;
        videoInfo::g_threadsMutex_resizedImageVector.lock();
        frame.image.ToHost();
        videoInfo::realImageVector.push_back(frame);
        videoInfo::resizedImageVector.push_back(resizedFrame);
        videoInfo::g_threadsMutex_resizedImageVector.unlock();
        // 然后再将原图pop出去
        videoInfo::g_threadsMutex_frameImageQueue.lock();
        videoInfo::frameImageQueue.pop();
        videoInfo::g_threadsMutex_frameImageQueue.unlock();
        // 计数
        resize_count++;
    }
    std::chrono::high_resolution_clock::time_point resize_end_time = std::chrono::high_resolution_clock::now();
    double_t resize_cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(resize_end_time - resize_start_time).count() / videoInfo::MS_PPE_SECOND;
    double_t resize_finish_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() / videoInfo::MS_PPE_SECOND;
    LogInfo << "总共" << target_count << "帧, 到缩放全部完成一共花费: " << resize_finish_time << ", 缩放本身花费: " << resize_cost_time;
}

// 推理线程
void inferMethod(std::chrono::high_resolution_clock::time_point start_time, size_t target_count)
{
    std::chrono::high_resolution_clock::time_point infer_start_time = std::chrono::high_resolution_clock::now();

    std::shared_ptr<MxBase::Model> modelDptr = std::make_shared<MxBase::Model>(videoInfo::modelPath, videoInfo::DEVICE_ID);

    size_t infer_count = 0;
    while (infer_count < target_count)
    {
        if (infer_count >= videoInfo::resizedImageVector.size())
        {
            continue;
        }
        // 从resize后的队列中取图片
        FrameImage resizedFrame = videoInfo::resizedImageVector[infer_count];
        std::vector<MxBase::Tensor> modelOutputs;
        MxBase::Tensor tensorImg = resizedFrame.image.ConvertToTensor();
        tensorImg.ToDevice(videoInfo::DEVICE_ID);
        std::vector<MxBase::Tensor> inputs;
        inputs.push_back(tensorImg);
        modelOutputs = modelDptr->Infer(inputs);
        for (auto output : modelOutputs)
        {
            output.ToHost();
        }

        // 将推理结果存入队列
        videoInfo::g_threadsMutex_inferOutputVector.lock();
        videoInfo::inferOutputVector.push_back(modelOutputs);
        videoInfo::g_threadsMutex_inferOutputVector.unlock();
        // 计数
        infer_count++;
    }
    std::chrono::high_resolution_clock::time_point infer_end_time = std::chrono::high_resolution_clock::now();
    double_t infer_cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(infer_end_time - infer_start_time).count() / videoInfo::MS_PPE_SECOND;
    double_t infer_finish_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() / videoInfo::MS_PPE_SECOND;
    LogInfo << "总共" << target_count << "帧, 到推理全部完成一共花费: " << infer_finish_time << ", 推理本身花费: " << infer_cost_time;
}

// 后处理线程
void postprocessMethod(std::chrono::high_resolution_clock::time_point start_time, size_t target_count)
{
    std::chrono::high_resolution_clock::time_point postprocess_start_time = std::chrono::high_resolution_clock::now();

    std::map<std::string, std::string> postConfig;
    postConfig.insert(std::pair<std::string, std::string>("postProcessConfigPath", videoInfo::configPath));
    postConfig.insert(std::pair<std::string, std::string>("labelPath", videoInfo::labelPath));
    std::shared_ptr<MxBase::Yolov3PostProcess> postProcessorDptr = std::make_shared<MxBase::Yolov3PostProcess>();
    if (postProcessorDptr == nullptr)
    {
        LogError << "init postProcessor failed, nullptr";
    }
    postProcessorDptr->Init(postConfig);

    size_t postprocess_count = 0;
    while (postprocess_count < target_count)
    {
        if (postprocess_count >= videoInfo::inferOutputVector.size())
        {
            continue;
        }
        // 取原图信息用于计算
        MxBase::Size originalSize = videoInfo::realImageVector[postprocess_count].image.GetOriginalSize();
        // 从推理结果的队列里面取出推理结果
        std::vector<MxBase::Tensor> modelOutputs = videoInfo::inferOutputVector[postprocess_count];
        FrameImage resizedFrame = videoInfo::resizedImageVector[postprocess_count];
        // 新的后处理过程
        MxBase::ResizedImageInfo imgInfo;
        auto shape = modelOutputs[0].GetShape();
        imgInfo.widthOriginal = originalSize.width;
        imgInfo.heightOriginal = originalSize.height;
        imgInfo.widthResize = videoInfo::YOLOV5_RESIZE;
        imgInfo.heightResize = videoInfo::YOLOV5_RESIZE;
        imgInfo.resizeType = MxBase::RESIZER_MS_KEEP_ASPECT_RATIO;
        // 因为yolov5要求输入图像为640*640，所以直接比较原图的height和width就好（如果不理解就去看cropResizePaste.hpp里的GetPasteRect函数）
        float resizeRate = originalSize.width > originalSize.height ? (originalSize.width * 1.0 / videoInfo::YOLOV5_RESIZE) : (originalSize.height * 1.0 / videoInfo::YOLOV5_RESIZE);
        imgInfo.keepAspectRatioScaling = 1 / resizeRate;
        std::vector<MxBase::ResizedImageInfo> imageInfoVec = {};
        imageInfoVec.push_back(imgInfo);
        // make postProcess inputs
        std::vector<MxBase::TensorBase> tensors;
        for (size_t i = 0; i < modelOutputs.size(); i++)
        {
            MxBase::MemoryData memoryData(modelOutputs[i].GetData(), modelOutputs[i].GetByteSize());
            MxBase::TensorBase tensorBase(memoryData, true, modelOutputs[i].GetShape(), MxBase::TENSOR_DTYPE_INT32);
            tensors.push_back(tensorBase);
        }
        // 后处理
        std::vector<std::vector<MxBase::ObjectInfo>> objectInfos;
        postProcessorDptr->Process(tensors, objectInfos, imageInfoVec);

        // 将后处理结果存入队列
        videoInfo::g_threadsMutex_postprocessOutputVector.lock();
        videoInfo::postprocessOutputVector.push_back(objectInfos);
        videoInfo::g_threadsMutex_postprocessOutputVector.unlock();
        // 计数
        postprocess_count++;
    }
    std::chrono::high_resolution_clock::time_point postprocess_end_time = std::chrono::high_resolution_clock::now();
    double_t postprocess_cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(postprocess_end_time - postprocess_start_time).count() / videoInfo::MS_PPE_SECOND;
    double_t postprocess_finish_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() / videoInfo::MS_PPE_SECOND;
    LogInfo << "总共" << target_count << "帧, 到后处理全部完成一共花费: " << postprocess_finish_time << ", 后处理本身花费: " << postprocess_cost_time;
}

// 跟踪去重线程
void trackMethod(std::chrono::high_resolution_clock::time_point start_time, size_t target_count)
{
    std::chrono::high_resolution_clock::time_point track_start_time = std::chrono::high_resolution_clock::now();

    std::shared_ptr<ascendVehicleTracking::MOTConnection> tracker0 = std::make_shared<ascendVehicleTracking::MOTConnection>();
    if (tracker0 == nullptr)
    {
        LogError << "init tracker0 failed, nullptr";
    }
    std::shared_ptr<ascendVehicleTracking::MOTConnection> tracker1 = std::make_shared<ascendVehicleTracking::MOTConnection>();
    if (tracker1 == nullptr)
    {
        LogError << "init tracker1 failed, nullptr";
    }
    // 用于计算帧率
    size_t old_count = 0;
    std::chrono::high_resolution_clock::time_point count_time;
    std::chrono::high_resolution_clock::time_point old_count_time = std::chrono::high_resolution_clock::now();
    size_t one_step = 2;
    size_t track_count = 0;
    while (track_count < target_count)
    {
        // 计算帧率
        // 如果count_time-old_count_time的值大于one_step，就计算一下这个step里面的帧数，然后除以step的值
        count_time = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(count_time - old_count_time).count() / videoInfo::MS_PPE_SECOND > one_step)
        {
            old_count_time = count_time;
            LogInfo << "rate: " << (track_count - old_count) / one_step * 1.0;
            old_count = track_count;
        }
        // 下面是业务循环
        if (track_count >= videoInfo::postprocessOutputVector.size())
        {
            continue;
        }
        // 从后处理结果的队列中取结果用于跟踪去重
        std::vector<std::vector<MxBase::ObjectInfo>> objectInfos = videoInfo::postprocessOutputVector[track_count];
        FrameImage frame = videoInfo::realImageVector[track_count];
        MxBase::Size originalSize = frame.image.GetOriginalSize();

        // 根据channelId的不同使用不同的tracker
        std::vector<MxBase::ObjectInfo> objInfos_ = {};
        if (frame.channelId == 0)
        {
            tracker0->ProcessSort(objectInfos, frame.frameId);
            APP_ERROR ret = tracker0->GettrackResult(objInfos_);
            if (ret != APP_ERR_OK)
            {
                LogError << "No tracker0";
            }
        }
        else
        {
            tracker1->ProcessSort(objectInfos, frame.frameId);
            APP_ERROR ret = tracker1->GettrackResult(objInfos_);
            if (ret != APP_ERR_OK)
            {
                LogError << "No tracker1";
            }
        }

        uint32_t video_height = originalSize.height;
        uint32_t video_width = originalSize.width;
        // 初始化OpenCV图像信息矩阵
        cv::Mat imgYuv = cv::Mat(video_height * videoInfo::YUV_BYTE_NU / videoInfo::YUV_BYTE_DE, video_width, CV_8UC1, frame.image.GetData().get());
        cv::Mat imgBgr = cv::Mat(video_height, video_width, CV_8UC3);
        // 颜色空间转换
        cv::cvtColor(imgYuv, imgBgr, cv::COLOR_YUV420sp2RGB);
        std::vector<MxBase::ObjectInfo> info;
        bool headFlag = false;
        for (uint32_t i = 0; i < objInfos_.size(); i++)
        {
            if (objInfos_[i].className == videoInfo::TARGET_CLASS_NAME)
            {
                headFlag = true;
                LogWarn << "Warning:Not wearing a helmet, channelId:" << frame.channelId << ", frameId:" << frame.frameId;
                // (blue, green, red)
                const cv::Scalar color = cv::Scalar(0, 0, 255);
                // width for rectangle
                const uint32_t thickness = 2;
                // draw the rectangle
                cv::rectangle(imgBgr,
                              cv::Rect(objInfos_[i].x0, objInfos_[i].y0, objInfos_[i].x1 - objInfos_[i].x0, objInfos_[i].y1 - objInfos_[i].y0),
                              color, thickness);
            }
        }
        // 如果检测结果中有head标签，就保存为图片
        if (headFlag)
        {
            // 把Mat类型的图像矩阵保存为图像到指定位置。
            std::string outPath = frame.channelId == 0 ? "one" : "two";
            std::string fileName = "./result/" + outPath + "/result" + std::to_string(frame.frameId) + ".jpg";
            cv::imwrite(fileName, imgBgr);
        }

        // 计数
        track_count++;
    }
    std::chrono::high_resolution_clock::time_point track_end_time = std::chrono::high_resolution_clock::now();
    double_t track_cost_time = std::chrono::duration_cast<std::chrono::milliseconds>(track_end_time - track_start_time).count() / videoInfo::MS_PPE_SECOND;
    double_t track_finish_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count() / videoInfo::MS_PPE_SECOND;
    LogInfo << "总共" << target_count << "帧, 到跟踪去重全部完成一共花费: " << track_finish_time << ", 跟踪去重本身花费: " << track_cost_time;
}

#endif
