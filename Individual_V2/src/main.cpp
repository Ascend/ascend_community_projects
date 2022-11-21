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

#include "fstream"
#include "chrono"
#include "time.h"
#include "AttrRecognition.h"
#include "ImageProcess.h"
#include "DetectAndAlign.h"
#include "opencv2/opencv.hpp"
#include "boost/filesystem.hpp"

using namespace MxBase;
namespace fs = boost::filesystem;

namespace
{
    const Size ATTR_RECOGNITION_SIZE(224, 224); // AttrRecognition's input size
    const Size YOLO_SIZE(416, 416);             // Yolo's input size
    const Size FACE_LANDMARK_SIZE(96, 96);      // FaceLandmark's input size
    const double_t CROP_EXPAND_RATIO = 0.2;     // Expand border when croping
    const uint32_t EVAL_INDEX_START = 182637;   // Start index of images to be tested;
}

// collect all images in datasetPath whose extension belongs to `extensions`
std::vector<cv::String> findAllImages(const fs::path &datasetPath)
{
    std::vector<fs::path> extensions = {"*.jpg", "*.jpeg", "*.JPEG", "*.JPG"};
    std::vector<cv::String> images;

    for (auto &extension : extensions)
    {
        std::vector<cv::String> temp;
        cv::String pattern = (datasetPath / extension).c_str();
        cv::glob(pattern, temp, false);

        // concatenate vectors
        images.insert(images.end(), temp.begin(), temp.end());
    }
    sort(images.begin(), images.end());

    std::vector<cv::String> subset;
    for (size_t i = 0; i < images.size(); i++)
    {
        // collect pictures with labels when calculating accuracy
        if (i >= EVAL_INDEX_START)
            subset.push_back(images[i]);
    }

    return subset;
}

// for only recognizing
APP_ERROR Recognize(const Image &alignedImage,
                    const std::shared_ptr<ImageProcess> &imageProcess,
                    const std::shared_ptr<AttrRecognition> attrRecognition,
                    std::vector<std::vector<ClassInfo>> &classInfos,
                    double_t &inferCostTime)
{
    // resize affined image for attr recognition's input
    Image recogInput;
    imageProcess->Resize(alignedImage, recogInput, ATTR_RECOGNITION_SIZE);

    // do attr recognition
    std::vector<Tensor> recogOutput;
    std::chrono::high_resolution_clock::time_point recogStart = std::chrono::high_resolution_clock::now();
    attrRecognition->Infer(recogInput, recogOutput);
    std::chrono::high_resolution_clock::time_point recogEnd = std::chrono::high_resolution_clock::now();
    inferCostTime = std::chrono::duration_cast<std::chrono::milliseconds>(recogEnd - recogStart).count();

    // get classinfos
    attrRecognition->PostProcess(recogOutput, classInfos);

    // print infer results
    for (size_t i = 0; i < classInfos.size(); i++)
    {
        for (auto &classInfo : classInfos[i])
        {
            LogInfo << "    " << classInfo.className << ":"
                    << classInfo.confidence;
        }
    }

    return APP_ERR_OK;
}

// detect face and align it,then recognize
APP_ERROR DetectAndRecognize(const std::string imgPath,
                             const std::shared_ptr<ImageProcess> &imageProcess,
                             const std::shared_ptr<Yolo> &yolo,
                             const std::shared_ptr<FaceLandMark> &faceLandmark,
                             const std::shared_ptr<AttrRecognition> &attrRecognition,
                             double_t &inferCostTime)
{
    // read image
    cv::Mat originalMat = cv::imread(imgPath);
    if (!originalMat.data)
    {
        LogError << "Can not read this image !";
        return APP_ERR_INVALID_FILE;
    }
    Image originalImg;
    imageProcess->ReadImage(imgPath, originalImg);

    // resize for yolo's input
    Image yoloInput;
    imageProcess->Resize(originalImg, yoloInput, YOLO_SIZE);
    cv::Mat resizedMat;
    cv::resize(originalMat, resizedMat, cv::Size(YOLO_SIZE.width, YOLO_SIZE.height));

    // do yolo infer
    std::vector<Tensor> yoloOutput;
    std::chrono::high_resolution_clock::time_point yoloStart = std::chrono::high_resolution_clock::now();
    yolo->Infer(yoloInput, yoloOutput);
    std::chrono::high_resolution_clock::time_point yoloEnd = std::chrono::high_resolution_clock::now();
    inferCostTime = std::chrono::duration_cast<std::chrono::milliseconds>(yoloEnd - yoloStart).count();

    // get crop config
    std::vector<Rect> cropConfigVec;
    yolo->PostProcess(yoloOutput, cropConfigVec);

    if (cropConfigVec.size() == 0)
    {
        LogInfo << "    "
                << "No face detected.";
    }
    else
    {
        // for every image to be croped
        for (size_t i = 0; i < cropConfigVec.size(); i++)
        {
            LogInfo << "    "
                    << i + 1 << " of " << cropConfigVec.size() << " faces :";
            auto cropConfig = cropConfigVec[i];

            // crop image
            cv::Mat cropedMat;
            Image cropedImage;
            imageProcess->Crop(resizedMat, cropedMat, cropConfig, CROP_EXPAND_RATIO);
            imageProcess->Crop(yoloInput, cropedImage, cropConfig, CROP_EXPAND_RATIO);

            // resize for face landmark
            Image landmarkInput;
            std::vector<Tensor> landmarkOutput;
            imageProcess->Resize(cropedImage, landmarkInput, FACE_LANDMARK_SIZE);

            // do face landmark detection
            std::chrono::high_resolution_clock::time_point landmarkStart = std::chrono::high_resolution_clock::now();
            faceLandmark->Infer(landmarkInput, landmarkOutput); // face landmark infer
            std::chrono::high_resolution_clock::time_point landmarkEnd = std::chrono::high_resolution_clock::now();
            inferCostTime += std::chrono::duration_cast<std::chrono::milliseconds>(landmarkEnd - landmarkStart).count();

            // do affine transform
            cv::Mat affinedMat;
            faceLandmark->PostProcess(landmarkOutput, cropedMat, affinedMat);

            // convert OpenCV's Mat to MxBase's Image
            Image alignedImage;
            imageProcess->ConvertMatToImage(affinedMat, alignedImage);

            // do attr recognition
            std::vector<std::vector<ClassInfo>> classInfos;
            double_t attrInferTime;
            Recognize(alignedImage, imageProcess, attrRecognition, classInfos, attrInferTime);
            inferCostTime += attrInferTime;
        }
    }
    return APP_ERR_OK;
}

void readArgs(int &argc, char *argv[], std::string &dataPath, std::string &imgPath, std::string &mode)
{
    // read param from command line
    int cmd = 0;
    const char *optstirng = "m:i:d:";
    while ((cmd = getopt(argc, argv, optstirng)) != -1)
    {
        switch (cmd)
        {
        case 'd':
            dataPath = std::string(optarg);
            break;
        case 'i':
            imgPath = std::string(optarg);
            break;
        case 'm':
            mode = std::string(optarg);
            if (mode != "eval" && mode != "run")
            {
                LogInfo << "Invalid mode value,should be `eval` or `run`.Use default value `run`.";
                mode = "eval";
            }
            break;
        default:
            break;
        }
    }
}

APP_ERROR CheckFile(const std::string &imgPath)
{
    // check imgPath
    if (!fs::exists(imgPath))
    {
        LogError << imgPath << " does not exist.Please check it.";
        return APP_ERR_COMM_NO_EXIST;
    }
    if (!fs::is_regular_file(imgPath))
    {
        LogError << imgPath << " is not a regular file.Please check it.";
        return APP_ERR_INVALID_FILE;
    }

    return APP_ERR_OK;
}

APP_ERROR CheckDir(const std::string &dataPath)
{
    // check dataPath
    if (!fs::exists(dataPath))
    {
        LogError << dataPath << " does not exist.Please check it.";
        return APP_ERR_COMM_NO_EXIST;
    }
    if (!fs::is_directory(dataPath))
    {
        LogError << dataPath << " is not a directory.Please check it.";
        return APP_ERR_INVALID_FILE;
    }
    if (fs::is_empty(dataPath))
    {
        LogError << dataPath << " is empty.Please check it.";
        return APP_ERR_INVALID_FILE;
    }

    return APP_ERR_OK;
}

void paramInit(V2Param &yoloParam, V2Param &faceLandmarkParam, V2Param &attrRecognitionParam, const uint32_t deviceId)
{
    yoloParam.configPath = "./models/yolov4.cfg";
    yoloParam.labelPath = "./models/coco.names";
    yoloParam.modelPath = "./models/yolov4_detection.om";
    yoloParam.deviceId = deviceId;

    faceLandmarkParam.configPath = "";
    faceLandmarkParam.labelPath = "";
    faceLandmarkParam.modelPath = "./models/face_quality_0605_b1.om";
    faceLandmarkParam.deviceId = deviceId;

    attrRecognitionParam.configPath = "./models/resnet50_aipp_tf.cfg";
    attrRecognitionParam.labelPath = "./models/attr.names";
    attrRecognitionParam.modelPath = "./models/Attribute_test.om";
    attrRecognitionParam.deviceId = deviceId;
}

int main(int argc, char *argv[])
{
    // ---------- params start---------------
    std::string mode = "run";                         // `run` for detecting and then recognizing;
                                                      // `eval` for only recognizing.
    std::string imgPath = "test.jpg";                 // used in run mode
    std::string dataPath = "CelebA/img_align_celeba"; // used in eval mode
    std::string outputFile = "infer_result.txt";
    const uint32_t deviceId = 0;

    readArgs(argc, argv, dataPath, imgPath, mode);

    V2Param yoloParam, faceLandmarkParam, attrRecognitionParam;
    paramInit(yoloParam, faceLandmarkParam, attrRecognitionParam, deviceId);

    // global init
    APP_ERROR ret;
    ret = MxInit();
    if (ret != APP_ERR_OK)
    {
        LogError << "MxInit failed, ret=" << ret << ".";
        return ret;
    }

    // model init
    auto imageProcess = std::make_shared<ImageProcess>(deviceId);
    auto attrRecognition = std::make_shared<AttrRecognition>(attrRecognitionParam);

    if (mode == "run")
    {
        ret = CheckFile(imgPath);
        if (ret != APP_ERR_OK)
            return ret;

        double_t e2eTime = 0;   // include resize,postprocess,infer,etc.
        double_t inferTime = 0; // only include model infer time

        auto yolo = std::make_shared<Yolo>(yoloParam);
        auto faceLandmark = std::make_shared<FaceLandMark>(faceLandmarkParam);

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        ret = DetectAndRecognize(imgPath, imageProcess, yolo, faceLandmark, attrRecognition, inferTime);
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        e2eTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        if (ret != APP_ERR_OK)
        {
            LogError << "Recognize failed.Please read infos above.";
            return APP_ERR_OK;
        }

        LogInfo << "In run mode:";
        LogInfo << "the picture's E2E total time is " << e2eTime << "ms";
        LogInfo << "the picture's model infer total time is " << inferTime << "ms";
    }
    else
    {
        ret = CheckDir(dataPath);
        if (ret != APP_ERR_OK)
            return ret;

        double_t e2eTime = 0;   // include resize,postprocess,infer,etc.
        double_t inferTime = 0; // only include model infer time
        std::ofstream out(outputFile, std::ios::out | std::ios::trunc);

        auto dataset = findAllImages(dataPath);
        if (dataset.size() == 0)
        {
            LogInfo << "`dataPath` has no valid image.Only JPEG format image is supported.";
            return APP_ERR_OK;
        }

        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < dataset.size(); i++)
        {
            LogInfo << "Processing " + std::to_string(i) + " of " + std::to_string(dataset.size());
            std::string path = dataset[i];

            Image alignedImage;
            imageProcess->ReadImage(path, alignedImage);

            double_t attrInferTime;
            std::vector<std::vector<ClassInfo>> classInfos;
            Recognize(alignedImage, imageProcess, attrRecognition, classInfos, attrInferTime);

            out << path;
            for (size_t i = 0; i < classInfos.size(); i++)
            {
                for (auto &classInfo : classInfos[i])
                    out << " " << classInfo.confidence;
            }
            out << std::endl;

            inferTime += attrInferTime;
        }
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        e2eTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        out.close();

        LogInfo << "In eval mode:";
        LogInfo << dataset.size() << " pictures's E2E total time is " << e2eTime << "ms";
        LogInfo << "E2E average time is " << e2eTime / dataset.size() << "ms";
        LogInfo << "Model infer average time is " << inferTime / dataset.size() << "ms";
    }
}