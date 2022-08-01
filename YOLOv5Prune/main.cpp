/*
# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 */

#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include "MxBase/Log/Log.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "opencv4/opencv2/opencv.hpp"
const int color_list[20][3] = { {216, 82, 24}, {236, 176, 31}, {125, 46, 141}, {118, 171, 47}, {76, 189, 237}, {238, 19, 46},
                            {76, 76, 76}, {153, 153, 153}, {255, 0, 0}, {255, 127, 0}, {190, 190, 0}, {0, 255, 0}, {0, 0, 255},
                            {170, 0, 255}, {84, 84, 0}, {84, 170, 0}, {84, 255, 0}, {170, 84, 0}, {170, 170, 0}, {170, 255, 0}}

float pad_w = 0.0, pad_h = 0.0;
float ratio = 1.0;


namespace {
APP_ERROR ReadFile(const std::string& filePath, MxStream::MxstDataInput& dataBuffer)
{
    char c[PATH_MAX + 1] = { 0x00 };
    size_t count = filePath.copy(c, PATH_MAX + 1);
    if (count != filePath.length()) {
        LogError << "Failed to copy file path(" << c << ").";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the absolute path of input file
    char path[PATH_MAX + 1] = { 0x00 };
    if ((strlen(c) > PATH_MAX) || (realpath(c, path) == nullptr)) {
        LogError << "Failed to get image, the image path is (" << filePath << ").";
        return APP_ERR_COMM_NO_EXIST;
    }
    // Open file with reading mode
    FILE *fp = fopen(path, "rb");
    if (fp == nullptr) {
        LogError << "Failed to open file (" << path << ").";
        return APP_ERR_COMM_OPEN_FAIL;
    }
    // Get the length of input file
    fseek(fp, 0, SEEK_END);
    long fileSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    // If file not empty, read it into FileInfo and return it
    if (fileSize > 0) {
        dataBuffer.dataSize = fileSize;
        dataBuffer.dataPtr = new (std::nothrow) uint32_t[fileSize];
        if (dataBuffer.dataPtr == nullptr) {
            LogError << "allocate memory with \"new uint32_t\" failed.";
            return APP_ERR_COMM_FAILURE;
        }

        uint32_t readRet = fread(dataBuffer.dataPtr, 1, fileSize, fp);
        if (readRet <= 0) {
            fclose(fp);
            delete dataBuffer.dataPtr;
            dataBuffer.dataPtr = nullptr;
            return APP_ERR_COMM_READ_FAIL;
        }
        fclose(fp);
        return APP_ERR_OK;
    }
    fclose(fp);
    return APP_ERR_COMM_FAILURE;
}

std::string ReadPipelineConfig(const std::string& pipelineConfigPath)
{
    std::ifstream file(pipelineConfigPath.c_str(), std::ifstream::binary);
    if (!file) {
        LogError << pipelineConfigPath <<" file dose not exist.";
        return "";
    }
    file.seekg(0, std::ifstream::end);
    uint32_t fileSize = file.tellg();
    file.seekg(0);
    auto dataPtr = new (std::nothrow) char[fileSize];
    if (dataPtr == nullptr) {
        LogError << GetError(APP_ERR_COMM_INIT_FAIL) << "The pointer is null.";
        return "";
    }
    std::unique_ptr<char[]> data(dataPtr);
    file.read(data.get(), fileSize);
    file.close();
    std::string pipelineConfig(data.get(), fileSize);
    return pipelineConfig;
}
}
struct Result
{
public:
    float x0, y0, x1, y1;
    std::string className;
    int classId;
    float conf;
};
cv::Mat letterBox(const cv::Mat& src)
{
	int in_w = src.cols;
	int in_h = src.rows;
	int tar_w = 512;
	int tar_h = 512;

	ratio = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
	int inside_w = std::round(in_w * ratio);
	int inside_h = std::round(in_h * ratio);
	pad_w = tar_w - inside_w;
    pad_h = tar_h - inside_h;
	cv::Mat resize_img;
	cv::resize(src, resize_img, cv::Size(inside_w, inside_h));
	cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
    const int div = 2;
	pad_w = pad_w / div;
	pad_h = pad_h / div;

	int topPad = int(std::round(pad_h - 0.1));
	int btmPad = int(std::round(pad_h + 0.1));
	int leftPad = int(std::round(pad_w - 0.1));
	int rightPad = int(std::round(pad_w + 0.1));
    int b = 0, g = 135, r = 0;
	cv::copyMakeBorder(resize_img, resize_img, topPad, btmPad, leftPad, rightPad, cv::BORDER_CONSTANT, cv::Scalar(b, g, r));
    cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);

	return resize_img;
}
std::vector<Result> ParseResult(const std::string& result)
{
    std::vector<Result> res;
    web::json::value jsonText = web::json::value::parse(result);
    if (jsonText.is_object()) {
        web::json::object textObject = jsonText.as_object();
        auto itInferObject = textObject.find("MxpiObject");
        if (itInferObject == textObject.end() || (!itInferObject->second.is_array())) {
            return {};
        }
        auto iter = itInferObject->second.as_array().begin();
        for (; iter != itInferObject->second.as_array().end(); iter++) {
            if (iter->is_object()) {
                Result tmp;
                auto modelInferObject = iter->as_object();
                auto it = modelInferObject.find("classVec");
                if (it != modelInferObject.end()) {
                    auto class_iter = it->second.as_array().begin();
                    if (class_iter->is_object()) {
                        auto classObject = class_iter->as_object();
                        auto class_it = classObject.find("className");
                        if (class_it != classObject.end()) {
                            tmp.className = class_it->second.as_string();
                        }
                        class_it = classObject.find("confidence");
                        if (class_it != classObject.end()) {
                            tmp.conf = float(class_it->second.as_double());
                        }
                        class_it = classObject.find("classId");
                        if (class_it != classObject.end()) {
                            tmp.classId = float(class_it->second.as_integer());
                        }
                    }
                }
                it = modelInferObject.find("x0");
                if (it != modelInferObject.end()) {
                    tmp.x0 = float(it->second.as_double());
                }
                it = modelInferObject.find("x1");
                if (it != modelInferObject.end()) {
                    tmp.x1 = float(it->second.as_double());
                }
                it = modelInferObject.find("y0");
                if (it != modelInferObject.end()) {
                    tmp.y0 = float(it->second.as_double());
                }
                it = modelInferObject.find("y1");
                if (it != modelInferObject.end()) {
                    tmp.y1 = float(it->second.as_double());
                }
                int topPad = int(std::round(pad_h - 0.1));
	            int leftPad = int(std::round(pad_w - 0.1));
                tmp.x0 = std::max((tmp.x0 - leftPad)/ratio, 0.0f);
                tmp.y0 = std::max((tmp.y0 - topPad)/ratio, 0.0f);
                tmp.x1 = (tmp.x1 - leftPad)/ratio;
                tmp.y1 = (tmp.y1 - topPad)/ratio;
       
                res.push_back(tmp);
            }
        }
    }
    return res;
}
void SaveImage(const std::string& result, const cv::Mat src, const std::string& line)
{
    auto res = ParseResult(result);
    for (auto it : res) {
        cv::Scalar color = cv::Scalar(color_list[it.classId][0], color_list[it.classId][1], color_list[it.classId][2]);
        cv::Rect rect(it.x0, it.y0, it.x1 - it.x0, it.y1 - it.y0);
        cv::rectangle(src, rect, color);
        char text[256];
        sprintf(text, "%s %.1f%", it.className.c_str(), it.conf);
        int baseLine = 0;
        double fontScale = 0.4;
        cv::Scalar fontColor = cv::Scalar(255, 255, 255);
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
        cv::rectangle(src, cv::Rect(cv::Point(it.x0, it.y0), cv::Size(label_size.width, label_size.height + baseLine)), color, -1);
        cv::putText(src, text, cv::Point(it.x0, it.y0 + label_size.height), cv::FONT_HERSHEY_SIMPLEX, fontScale, fontColor);
    }
    cv::imwrite("./image_result/"+line+".jpg", src);
}
void SaveTxt(const std::string& result, const std::string& line)
{
    // web::json::value jsonText = web::json::value::parse(result);
    auto res = ParseResult(result);
    for (auto it : res) {
        std::ofstream outfile("./txt_result/det_test_" + it.className + ".txt", std::ios::app);
        char text[256];
        sprintf(text, "%s %f %f %f %f %f\n", line.c_str(), it.conf, it.x0, it.y0, it.x1, it.y1);
        outfile<<text;
        outfile.close();
    }
}

double time_min = DBL_MAX;
double time_max = -DBL_MAX;
double time_avg = 0;
long loop_num = 0;
std::string pipelineConfigPath = "";
const std::string streamName = "detection";
std::string task;
std::string imageSetFile;
std::string imageSetPath;
bool saveImage = false, saveTxt = false;
int inPluginId = 0;

int run()
{
    std::ifstream in(imageSetFile);
    std::string line;
    // read pipeline config file
    std::string pipelineConfig = ReadPipelineConfig(pipelineConfigPath);
    if (pipelineConfig == "") {
        LogError << "Read pipeline failed.";
        return APP_ERR_COMM_INIT_FAIL;
    }
    MxStream::MxStreamManager mxStreamManager;
    // init stream manager
    APP_ERROR ret = mxStreamManager.InitManager();
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to init Stream manager.";
        return ret;
    }
    // create stream by pipeline config file
    ret = mxStreamManager.CreateMultipleStreams(pipelineConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to create Stream.";
        return ret;
    }
    if (in) {
        while (getline(in, line)) {
            loop_num++;
            std::string img_path = imageSetPath+'/'+line+".jpg";
            MxStream::MxstDataInput dataBuffer;
            cv::Mat src;
            auto start = clock();
            if (task == "eval") {
                src = cv::imread(img_path);
                cv::Mat img = letterBox(src);
                cv::imwrite("./tmp.jpg", img);
                ret = ReadFile("./tmp.jpg", dataBuffer);
            }else if (task == "detect") {
                src = cv::imread(img_path);
                ret = ReadFile(img_path, dataBuffer);
            }else {
                ret = ReadFile(img_path, dataBuffer);
            }
            if (ret != APP_ERR_OK) {
                LogError << GetError(ret) << "Failed to read image file.";
                return ret;
            }
            // send data into stream
            ret = mxStreamManager.SendData(streamName, inPluginId, dataBuffer);
            if (ret != APP_ERR_OK) {
                LogError << GetError(ret) << "Failed to send data to stream.";
                delete dataBuffer.dataPtr;
                dataBuffer.dataPtr = nullptr;
                return ret;
            }
            // get stream output
            MxStream::MxstDataOutput* output = mxStreamManager.GetResult(streamName, inPluginId);
            if (output == nullptr) {
                LogError << "Failed to get pipeline output.";
                delete dataBuffer.dataPtr;
                dataBuffer.dataPtr = nullptr;
                return ret;
            }
            double time =  (double)(clock() - start) / CLOCKS_PER_SEC;
            time_min = (std::min)(time_min, time);
            time_max = (std::max)(time_max, time);
            time_avg += time;
            std::string result = std::string((char *)output->dataPtr, output->dataSize);
            if (saveImage == true) { SaveImage(result, src, line); }      
            if (saveTxt == true) { SaveTxt(result, line); }
            delete output;
            delete dataBuffer.dataPtr;
            dataBuffer.dataPtr = nullptr;
        }
        time_avg /= loop_num;
    }
    in.close();
    mxStreamManager.DestroyAllStreams();
    return 0;
}
int main(int argc, char* argv[])
{
    int index = 1;
    task = argv[index++];
    imageSetFile = argv[index++];
    imageSetPath = argv[index];
    if (task == "eval") {
        pipelineConfigPath = "pipeline/eval.pipeline";
    }else if (task == "speed" || task == "detect") {
        pipelineConfigPath = "pipeline/detect.pipeline";
    }else {
        std::cout<<"Undefined task!"<<std::endl;
        return 1;
    }
    if (task == "eval") { saveTxt = true; }
    if (task == "detect") { saveImage = true; }

    if (saveImage) { system("rm -rf image_result && mkdir image_result"); }
    if (saveTxt) { system("rm -rf txt_result && mkdir txt_result"); }
    int ret = run();
    if (ret != 0) {
        std::cout<<"Failed to run"<<std::endl;
        return 1;
    }
    char msg[256];
    const int millisecondPerSec = 1000;
    sprintf(msg, "image count = %ld \nmin = %.2fms  max = %.2fms  avg = %.2fms \navg fps = %.2f fps\n", loop_num, time_min*millisecondPerSec, time_max*millisecondPerSec, time_avg*millisecondPerSec, millisecondPerSec/(time_avg*millisecondPerSec));
    std::cout << "时间统计：\n";
    std::cout << msg;

    return 0;
}