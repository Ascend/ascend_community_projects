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

#include "chrono"
#include "Mono.h"
#include "boost/filesystem.hpp"
#include "boost/algorithm/string.hpp"

using namespace MxBase;
namespace fs = boost::filesystem;

// collect all images in datasetPath whose extension belongs to `extensions`
std::vector<cv::String> findAllImages(const fs::path datasetPath)
{
    std::vector<cv::String> images;

    std::vector<fs::path> extensions = {"*.jpg", "*.jpeg", "*.JPEG", "*.JPG"};

    for (auto &extension : extensions)
    {
        std::vector<cv::String> temp;
        cv::String pattern = (datasetPath / extension).c_str();
        cv::glob(pattern, temp, false);

        // concatenate vectors
        images.insert(images.end(), temp.begin(), temp.end());
    }

    std::sort(images.begin(), images.end());
    return images;
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

void readArgs(int &argc, char *argv[], std::string &datasetPath, std::string &imagePath, std::string &mode, std::string &outputPath)
{
    int cmd = 0;
    const char *optstirng = "m:i:d:o:";
    while ((cmd = getopt(argc, argv, optstirng)) != -1)
    {
        switch (cmd)
        {
        case 'd':
            datasetPath = std::string(optarg);
            break;
        case 'i':
            imagePath = std::string(optarg);
            break;
        case 'm':
            mode = std::string(optarg);
            if (mode != "eval" && mode != "run")
            {
                std::cout << "Invalid mode value,should be `eval` or `run`.Use default value `run`." << std::endl;
                mode = "eval";
            }
            break;
        case 'o':
            outputPath = std::string(optarg);
            break;
        default:
            break;
        }
    }
}

int main(int argc, char *argv[])
{
    const uint32_t deviceID = 0;
    const std::string modelPath = "models/AdaBins_nyu.om";

    std::string imagePath = "test.jpg";
    std::string datasetPath = "dataset";
    std::string outputPath = "results";
    std::string mode = "run"; // set `run` to run and save visualized pictures.
                              // set `eval` to eval and save binary files for calculating accuray

    readArgs(argc, argv, datasetPath, imagePath, mode, outputPath);

    // init param
    V2Param v2Param(deviceID, modelPath);

    // init model
    auto depthEstimation = std::make_shared<DepthEstimation>(v2Param);

    // check outputPath
    if (!fs::exists(outputPath))
    {
        LogInfo << "`outputPath` " << outputPath << " does not exist.Please create it.";
        return APP_ERR_OK;
    }
    if (!fs::is_directory(outputPath))
    {
        LogInfo << "`outputPath` " << outputPath << " is not a directory.Please check it.";
        return APP_ERR_OK;
    }

    std::vector<cv::String> images;
    std::string saveExtension;
    if (mode == "run")
    {
        APP_ERROR ret = CheckFile(imagePath);
        if (ret != APP_ERR_OK)
            return ret;

        images.push_back(imagePath);
        saveExtension = ".jpg";
    }
    else
    {
        APP_ERROR ret = CheckDir(datasetPath);
        if (ret != APP_ERR_OK)
            return ret;

        // collect all JPEG format images
        images = findAllImages(datasetPath);
        saveExtension = ".tiff";
    }

    if (images.size() == 0)
    {
        LogError << "Dataset is empty!Only JPEG format pictures will be processed.Please check it.";
        return APP_ERR_COMM_NO_EXIST;
    }

    // process every image
    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < images.size(); i++)
    {
        LogInfo << "Processing " << i + 1 << " of " << images.size() << " pictures";

        auto imgPath = images[i];

        cv::Mat originalMat = cv::imread(imgPath);
        if (!originalMat.data)
        {
            LogError << "Can not read this image ! --- " << imgPath;
            return APP_ERR_INVALID_FILE;
        }

        // decode image
        Image decodedImage;
        depthEstimation->ReadImage(imgPath, decodedImage);

        // resize image
        Image resizedImage;
        depthEstimation->Resize(decodedImage, resizedImage);

        // do infer
        std::vector<Tensor> outputs;
        depthEstimation->Infer(resizedImage, outputs);

        // do postprocess
        cv::Mat outputImage;
        depthEstimation->PostProcess(outputImage, outputs, mode);

        // save infer reuslt
        fs::path stem = ((fs::path)imgPath).stem();
        fs::path filename = (std::string)stem.c_str() + saveExtension;
        filename = (fs::path)outputPath / filename;
        cv::imwrite(filename.c_str(), outputImage);
    }
    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
    double_t e2eTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    LogInfo << "In " << mode << " mode:";
    LogInfo << images.size() << " pictures's E2E total time is " << e2eTime << "ms";
    LogInfo << "E2E average time is " << e2eTime / images.size() << "ms";
}
