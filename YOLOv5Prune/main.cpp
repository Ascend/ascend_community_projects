/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: Complete Sample Implementation of Target Detection in C++.
 * Author: MindX SDK
 * Create: 2021
 * History: NA
 */

#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include "MxBase/Log/Log.h"
#include "MxStream/StreamManager/MxStreamManager.h"
#include "opencv4/opencv2/opencv.hpp"
#include "color.h"


float pad_w, pad_h;
float ratio;


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
struct Result{
    public:
        float x0,y0,x1,y1;     //box
        std::string class_name;
        int class_id;
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
	cv::resize(src, resize_img, cv::Size(inside_w, inside_h));  //最小的Resize
	cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);
	pad_w = pad_w / 2;
	pad_h = pad_h / 2;

	int topPad = int(std::round(pad_h - 0.1));
	int btmPad = int(std::round(pad_h + 0.1));
	int leftPad = int(std::round(pad_w - 0.1));
	int rightPad = int(std::round(pad_w + 0.1));

	cv::copyMakeBorder(resize_img, resize_img, topPad, btmPad, leftPad, rightPad, cv::BORDER_CONSTANT, cv::Scalar(0, 135, 0));
    cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2RGB);

	return resize_img;
}
std::vector<Result> ParseResult(const std::string& result){
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
                    if(class_iter->is_object()){
                        auto classObject = class_iter->as_object();
                        auto class_it = classObject.find("className");
                        if (class_it != classObject.end()) {
                            tmp.class_name = class_it->second.as_string();
                        }
                        class_it = classObject.find("confidence");
                        if (class_it != classObject.end()) {
                            tmp.conf = float(class_it->second.as_double());
                        }
                        class_it = classObject.find("classId");
                        if (class_it != classObject.end()) {
                            tmp.class_id = float(class_it->second.as_integer());
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
                tmp.x0 = std::max((tmp.x0 - leftPad)/ratio , 0.0f);
                tmp.y0 = std::max((tmp.y0 - topPad)/ratio, 0.0f);
                tmp.x1 = (tmp.x1 - leftPad)/ratio;
                tmp.y1 = (tmp.y1 - topPad)/ratio;    
       
                res.push_back(tmp);
            } 
        }    
    }
    return res;
}
void SaveImage(const std::string& result, const cv::Mat src,const std::string& line){
    // web::json::value jsonText = web::json::value::parse(result);
    auto res = ParseResult(result);
    for(auto it : res){
        cv::Scalar color = cv::Scalar(color_list[it.class_id][0], color_list[it.class_id][1], color_list[it.class_id][2]);
        cv::Rect rect(it.x0, it.y0, it.x1 - it.x0, it.y1 - it.y0);
        cv::rectangle(src, rect, color);
        char text[256];
        sprintf(text, "%s %.1f%%", it.class_name.c_str(), it.conf * 100);
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
        cv::rectangle(src, cv::Rect(cv::Point(it.x0, it.y0), cv::Size(label_size.width, label_size.height + baseLine)), color, -1);
        cv::putText(src, text, cv::Point(it.x0, it.y0 + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }
    cv::imwrite("./image_result/"+line+".jpg", src);  
}
void SaveTxt(const std::string& result, const std::string& line){
    // web::json::value jsonText = web::json::value::parse(result);
    auto res = ParseResult(result);
    for(auto it : res){
        std::ofstream outfile("./txt_result/det_test_" + it.class_name + ".txt", std::ios::app);
        char text[256];
        sprintf(text, "%s %f %f %f %f %f\n", line.c_str(), it.conf, it.x0, it.y0, it.x1, it.y1);
        outfile<<text;
        outfile.close();
    }
 
}

int main(int argc, char* argv[])
{
    bool save_image = false, save_txt = true, speed = false;

    const std::string imagesetfile = "/home/wangshengke3/VOCdevkit/VOC2007/ImageSets/Main/test.txt";
    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;
    long loop_num = 0;

    std::ifstream in(imagesetfile);
    std::ofstream *outfile;
    std::string line;

    if(save_image){
        system("rm -rf image_result && mkdir image_result");
    }
    if(save_txt){
        system("rm -rf txt_result && mkdir txt_result");
    }

    int inPluginId = 0;
    std::string pipelineConfigPath = "pipeline/Sample.pipeline";
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

    if(in){

        while(getline(in, line)){
            loop_num++;

            std::string streamName = "detection";
            std::string img_path = "/home/wangshengke3/VOCdevkit/VOC2007/JPEGImages/"+line+".jpg";

            cv::Mat src = cv::imread(img_path);
            cv::Mat img = letterBox(src);
            cv::imwrite("./tmp.jpg", img);
  
            MxStream::MxstDataInput dataBuffer;
            ret = ReadFile("./tmp.jpg", dataBuffer);

            if (ret != APP_ERR_OK) {
                LogError << GetError(ret) << "Failed to read image file.";
                return ret;
            }
            auto start = clock();
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
            // LogInfo <<"Results:" << result;

            if(save_image == true)
                SaveImage(result, src, line);
            if(save_txt == true)
                SaveTxt(result, line);

            delete output;    // destroy streams
            delete dataBuffer.dataPtr;
            dataBuffer.dataPtr = nullptr;                         
        }
        time_avg /= loop_num;
        char msg[256];
        sprintf(msg,"image count = %ld\n min = %.2fms  max = %.2fms  avg = %.2fms \n avg fps = %.2f", loop_num, time_min *1000, time_max*1000, time_avg*1000, 1000/time_max*1000);
        LogInfo<<"推理时间统计：";
        LogInfo<<msg;
    }

    in.close();
    mxStreamManager.DestroyAllStreams();                   

    return 0;
}