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

#include "utils.h"

extern "C"
{
#include "libavformat/avformat.h"
}

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

#include <time.h>
#include <sys/stat.h>
using namespace std;
using namespace videoInfo;
namespace frameConfig
{
    size_t channelId0 = 0;
    size_t channelId1 = 1;
    size_t frameCountChannel0 = 300;
    size_t frameCountChannel1 = 300;
    size_t skipIntervalChannel0 = 2;
    size_t skipIntervalChannel1 = 2;

    int arg_one_video = 4;
    int arg_two_video = 7;
    int arg1 = 1;
    int arg2 = 2;
    int arg3 = 3;
    int arg4 = 4;
    int arg5 = 5;
    int arg6 = 6;
    int arg7 = 7;

    // channel0对应文件的指针
    AVFormatContext *pFormatCtx0 = nullptr;
    // channel1对应文件的指针
    AVFormatContext *pFormatCtx1 = nullptr;
}

// ffmpeg拉流
AVFormatContext *CreateFormatContext(std::string filePath)
{
    LogInfo << "start to CreatFormatContext!";
    // creat message for stream pull
    AVFormatContext *formatContext = nullptr;
    AVDictionary *options = nullptr;

    LogInfo << "start to avformat_open_input!";
    int ret = avformat_open_input(&formatContext, filePath.c_str(), nullptr, &options);
    if (options != nullptr)
    {
        av_dict_free(&options);
    }
    if (ret != 0)
    {
        LogError << "Couldn't open input stream " << filePath.c_str() << " ret=" << ret;
        return nullptr;
    }
    ret = avformat_find_stream_info(formatContext, nullptr);
    if (ret != 0)
    {
        LogError << "Couldn't open input stream information";
        return nullptr;
    }
    return formatContext;
}

// 真正的拉流函数
void PullStream0(std::string filePath)
{
    av_register_all();
    avformat_network_init();
    frameConfig::pFormatCtx0 = avformat_alloc_context();
    frameConfig::pFormatCtx0 = CreateFormatContext(filePath);
    av_dump_format(frameConfig::pFormatCtx0, 0, filePath.c_str(), 0);
}
void PullStream1(std::string filePath)
{
    av_register_all();
    avformat_network_init();
    frameConfig::pFormatCtx1 = avformat_alloc_context();
    frameConfig::pFormatCtx1 = CreateFormatContext(filePath);
    av_dump_format(frameConfig::pFormatCtx1, 0, filePath.c_str(), 0);
}

// 视频解码回调(样例代码，测试可以跑通，但是不能直接复用)
APP_ERROR CallBackVdec(MxBase::Image &decodedImage, uint32_t channelId, uint32_t frameId, void *userData)
{
    FrameImage frameImage;
    frameImage.image = decodedImage;
    frameImage.channelId = channelId;
    frameImage.frameId = frameId;

    videoInfo::g_threadsMutex_frameImageQueue.lock();
    videoInfo::frameImageQueue.push(frameImage);
    videoInfo::g_threadsMutex_frameImageQueue.unlock();

    return APP_ERR_OK;
}

// 获取H264中的帧
void GetFrame(AVPacket &pkt, FrameImage &frameImage, AVFormatContext *pFormatCtx)
{
    av_init_packet(&pkt);
    int ret = av_read_frame(pFormatCtx, &pkt);
    if (ret != 0)
    {
        LogInfo << "[StreamPuller] channel Read frame failed, continue!";
        if (ret == AVERROR_EOF)
        {
            LogInfo << "[StreamPuller] channel StreamPuller is EOF, over!";
            return;
        }
        return;
    }
    else
    {
        if (pkt.size <= 0)
        {
            LogError << "Invalid pkt.size: " << pkt.size;
            return;
        }

        // send to the device
        auto hostDeleter = [](void *dataPtr) -> void {};
        MxBase::MemoryData data(pkt.size, MxBase::MemoryData::MEMORY_HOST);
        MxBase::MemoryData src((void *)(pkt.data), pkt.size, MxBase::MemoryData::MEMORY_HOST);
        APP_ERROR ret = MxBase::MemoryHelper::MxbsMallocAndCopy(data, src);
        if (ret != APP_ERR_OK)
        {
            LogError << "MxbsMallocAndCopy failed!";
        }
        std::shared_ptr<uint8_t> imageData((uint8_t *)data.ptrData, hostDeleter);

        MxBase::Image subImage(imageData, pkt.size);
        frameImage.image = subImage;

        LogDebug << "'channelId = " << frameImage.channelId << ", frameId = " << frameImage.frameId << " , dataSize = " << frameImage.image.GetDataSize();

        av_packet_unref(&pkt);
    }
    return;
}

// 视频流解码线程 frameCount:要求遍历的帧的总数 skipInterval:跳帧的间隔
void VdecThread0(size_t frameCount, size_t skipInterval, int32_t channelId, uint32_t src_width, uint32_t src_height)
{
    AVPacket pkt;
    uint32_t frameId = 0;
    // 解码器参数
    MxBase::VideoDecodeConfig config;
    MxBase::VideoDecodeCallBack cPtr = CallBackVdec;
    config.width = src_width;
    config.height = src_height;
    config.callbackFunc = cPtr;
    // 跳帧控制
    config.skipInterval = skipInterval;

    std::shared_ptr<MxBase::VideoDecoder> videoDecoder = std::make_shared<MxBase::VideoDecoder>(config, videoInfo::DEVICE_ID, channelId);
    for (size_t i = 0; i < frameCount; i++)
    {
        MxBase::Image subImage;
        FrameImage frame;
        frame.channelId = channelId;
        frame.frameId = frameId;
        frame.image = subImage;
        GetFrame(pkt, frame, frameConfig::pFormatCtx0);
        APP_ERROR ret = videoDecoder->Decode(frame.image.GetData(), frame.image.GetDataSize(), frameId, &videoInfo::frameImageQueue);
        if (ret != APP_ERR_OK)
        {
            LogError << "videoDecoder Decode failed. ret is: " << ret;
        }
        frameId += 1;
    }
}

// 视频流解码线程 frameCount:要求遍历的帧的总数 skipInterval:跳帧的间隔
void VdecThread1(size_t frameCount, size_t skipInterval, int32_t channelId, uint32_t src_width, uint32_t src_height)
{
    AVPacket pkt;
    uint32_t frameId = 0;
    // 解码器参数
    MxBase::VideoDecodeConfig config;
    MxBase::VideoDecodeCallBack cPtr = CallBackVdec;
    config.width = src_width;
    config.height = src_height;
    config.callbackFunc = cPtr;
    // 跳帧控制
    config.skipInterval = skipInterval;

    std::shared_ptr<MxBase::VideoDecoder> videoDecoder = std::make_shared<MxBase::VideoDecoder>(config, videoInfo::DEVICE_ID, channelId);
    for (size_t i = 0; i < frameCount; i++)
    {
        MxBase::Image subImage;
        FrameImage frame;
        frame.channelId = channelId;
        frame.frameId = frameId;
        frame.image = subImage;
        GetFrame(pkt, frame, frameConfig::pFormatCtx1);
        APP_ERROR ret = videoDecoder->Decode(frame.image.GetData(), frame.image.GetDataSize(), frameId, &videoInfo::frameImageQueue);
        if (ret != APP_ERR_OK)
        {
            LogError << "videoDecoder Decode failed. ret is: " << ret;
        }
        frameId += 1;
    }
}

// 检查文件（是否存在、是否为文件夹）
bool checkFile(std::string &fileName)
{
    // 判断视频文件是否存在
    ifstream f(fileName.c_str());
    if (!f.good())
    {
        LogError << "file not exists! " << fileName;
        return false;
    }
    else
    {
        // 如果存在还需要判断是否是文件夹
        struct stat s;
        if (stat(fileName.c_str(), &s) == 0)
        {
            if (s.st_mode & S_IFDIR)
            {
                LogError << fileName << " is a directory!";
                return false;
            }
        }
    }
    return true;
}

bool checkArg(int argc, char *argv[])
{
    // 检测是否输入了文件路径
    if (argc <= 1)
    {
        LogWarn << "Please input video path, such as './video_sample test.264'.";
        return false;
    }
    else if(argc == arg_one_video)
    {
        std::string videoPath0 = argv[arg1];
        if(!checkFile(videoPath0))
        {
            return false;
        }
        uint32_t src_width0 = (uint32_t)stoul(argv[arg2]);
        uint32_t src_height0 = (uint32_t)stoul(argv[arg3]);
        LogWarn << "videoPath0: " << videoPath0 << ", src_width0: " << src_width0 << ", src_height0: " << src_height0;
        PullStream0(videoPath0);
        std::thread threadVdec0(VdecThread0, frameConfig::frameCountChannel0, frameConfig::skipIntervalChannel0, frameConfig::channelId0, src_width0, src_height0);
        threadVdec0.join();
    }
    else if(argc == arg_two_video)
    {
        std::string videoPath0 = argv[arg1];
        std::string videoPath1 = argv[arg4];
        if(!checkFile(videoPath0))
        {
            return false;
        }
        if(!checkFile(videoPath1))
        {
            return false;
        }

        uint32_t src_width0 = (uint32_t)stoul(argv[arg2]);
        uint32_t src_height0 = (uint32_t)stoul(argv[arg3]);
        LogWarn << "videoPath0: " << videoPath0 << ", src_width0: " << src_width0 << ", src_height0: " << src_height0;
        PullStream0(videoPath0);
        std::thread threadVdec0(VdecThread0, frameConfig::frameCountChannel0, frameConfig::skipIntervalChannel0, frameConfig::channelId0, src_width0, src_height0);
        threadVdec0.join();

        uint32_t src_width1 = (uint32_t)stoul(argv[arg5]);
        uint32_t src_height1 = (uint32_t)stoul(argv[arg6]);
        LogWarn << "videoPath1: " << videoPath1 << ", src_width1: " << src_width1 << ", src_height1: " << src_height1;
        PullStream1(videoPath1);
        std::thread threadVdec1(VdecThread1, frameConfig::frameCountChannel1, frameConfig::skipIntervalChannel1, frameConfig::channelId1, src_width1, src_height1);
        threadVdec1.join();
    }
    else
    {
        LogWarn << "usage: ./main test1.264 test1_width test1_height [test2.264 test2_width test2_height]";
        return false;
    }
    return true;
}

APP_ERROR main(int argc, char *argv[])
{
    // 初始化
    APP_ERROR ret = MxBase::MxInit();
    if (ret != APP_ERR_OK)
    {
        LogError << "MxInit failed, ret=" << ret << ".";
    }
    if(!checkArg(argc, argv))
    {
        return APP_ERR_OK;
    }

    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
    size_t target_count = videoInfo::frameImageQueue.size();

    // resize线程
    std::thread resizeThread(resizeMethod, start_time, target_count);
    resizeThread.join();

    // 推理线程
    std::thread inferThread(inferMethod, start_time, target_count);
    inferThread.join();

    // 后处理线程
    std::thread postprocessThread(postprocessMethod, start_time, target_count);
    postprocessThread.join();

    // 跟踪去重线程
    std::thread trackThread(trackMethod, start_time, target_count);
    trackThread.join();

    return APP_ERR_OK;
}