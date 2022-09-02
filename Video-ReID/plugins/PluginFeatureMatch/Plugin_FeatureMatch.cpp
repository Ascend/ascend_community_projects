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

#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"
#include "Plugin_FeatureMatch.h"
#include <algorithm>
#include <iostream>
#include <fstream>


using namespace MxPlugins;
using namespace MxTools;
using namespace MxBase;
using namespace cv;


APP_ERROR PluginFeatureMatch::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap) {
    LogInfo << "Begin to initialize PluginFeatureMatch(" << pluginName_ << ").";
    // Get the property values by key
    std::shared_ptr<std::string> querySource = std::static_pointer_cast<std::string>(configParamMap["querySource"]);
    querySource_ = *querySource;
    std::shared_ptr<std::string> objectSource = std::static_pointer_cast<std::string>(configParamMap["objectSource"]);
    objectSource_ = *objectSource;
    std::shared_ptr<std::string> galleryFeaturesPath = std::static_pointer_cast<std::string>(configParamMap["galleryFeaturesPath"]);
    galleryFeaturesPath_ = *galleryFeaturesPath;
    std::shared_ptr<std::string> galleryIdsPath = std::static_pointer_cast<std::string>(configParamMap["galleryIdsPath"]);
    galleryIdsPath_ = *galleryIdsPath;
    std::shared_ptr<std::string> metric = std::static_pointer_cast<std::string>(configParamMap["metric"]);
    metric_ = *metric;
    
    std::shared_ptr<std::float_t> threshold = std::static_pointer_cast<float_t>(configParamMap["threshold"]);
    threshold_ = *threshold;
    
    ReadGalleryFeatures(galleryFeaturesPath_, galleryIdsPath_, galleryFeatures, galleryIds);
    LogInfo << "End to initialize PluginFeatureMatch(" << pluginName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR PluginFeatureMatch::DeInit() {
    LogInfo << "Begin to deinitialize PluginFeatureMatch(" << pluginName_ << ").";
    LogInfo << "End to deinitialize PluginFeatureMatch(" << pluginName_ << ").";
    return APP_ERR_OK;
}

APP_ERROR PluginFeatureMatch::CheckDataSource(MxTools::MxpiMetadataManager &mxpiMetadataManager,
                                              MxTools::MxpiMetadataManager &mxpiMetadataManager1) {
    if (mxpiMetadataManager.GetMetadata(querySource_) == nullptr) {
        LogDebug << GetError(APP_ERR_METADATA_IS_NULL, pluginName_)
        << "class metadata is null. please check"
                 << "Your property querySource(" << querySource_ << ").";
        return APP_ERR_METADATA_IS_NULL;
    }
    if (mxpiMetadataManager1.GetMetadata(objectSource_) == nullptr) {
        LogDebug << GetError(APP_ERR_METADATA_IS_NULL, pluginName_)
        << "class metadata is null. please check"
                 << "Your property objectSource(" << objectSource_ << ").";
        return APP_ERR_METADATA_IS_NULL;
    }
    return APP_ERR_OK;
}

void EuclideanSquaredDistance(Mat queryFeatures, Mat galleryFeatures, Mat &distMat) {
    int admm_beta = 1;
    int admm_alpha = -2;
    int square = 2;
    Mat qdst1;
    pow(queryFeatures, square, qdst1);
    Mat qdst2;
    reduce(qdst1, qdst2, 1, REDUCE_SUM);
    
    Mat qdst3 = Mat(qdst2.rows, galleryFeatures.rows, CV_32FC1);
    for (int i = 0; i < galleryFeatures.rows; i++) {
        Mat dstTemp = qdst3.col(i);
        qdst2.copyTo(dstTemp);
    }
     
    Mat gdst1;
    pow(galleryFeatures, square, gdst1);
    Mat gdst2;
    reduce(gdst1, gdst2, 1, REDUCE_SUM);

    Mat gdst3 = Mat(gdst2.rows, queryFeatures.rows, CV_32FC1);
    for (int i = 0; i < queryFeatures.rows;i++) {
        Mat dstTemp = gdst3.col(i);
        gdst2.copyTo(dstTemp);
    }
    
    Mat gdst4;
    transpose(gdst3, gdst4);
    Mat dst1 = qdst3 +gdst4;

    Mat gdst5;
    transpose(galleryFeatures, gdst5);
    Mat dst2 = queryFeatures * gdst5;

    distMat = admm_beta * dst1 + admm_alpha * dst2;
}

void CosineDistance(Mat queryFeatures, Mat galleryFeatures, Mat &distMat) {
    Mat gdst1;
    
    transpose(galleryFeatures, gdst1);
    distMat = queryFeatures * gdst1;
}

APP_ERROR PluginFeatureMatch::ComputeDistance(MxpiTensorPackageList queryTensors,
                                              Mat galleryFeatures, int tensorSize, Mat &distMat) {
    Mat queryFeatures = Mat::zeros(tensorSize, queryTensors.
        tensorpackagevec(0).tensorvec(0).tensorshape(1), CV_32FC1);

    for (int i = 0; i < tensorSize; i++) {
        Mat dstTemp = queryFeatures.row(i);
        auto vec = queryTensors.tensorpackagevec(i).tensorvec(0);
        Mat feature = Mat(vec.tensorshape(0), vec.tensorshape(1), CV_32FC1,
            (void *) vec.tensordataptr());
        feature.copyTo(dstTemp);
    }
    if (!(metric_.compare("euclidean"))) {
        EuclideanSquaredDistance(queryFeatures, galleryFeatures, distMat);
    }
    else if (!(metric_.compare("cosine"))) {
        CosineDistance(queryFeatures, galleryFeatures, distMat);
    }
    return APP_ERR_OK;
}

void PluginFeatureMatch::GenerateOutput(MxpiObjectList srcObjectList, Mat distMat,
                                        int tensorSize, std::vector<std::string> galleryIds,
                                        MxpiObjectList &dstMxpiObjectList) {
    std::vector<float> minValues = {};
    for (int i = 0; i < tensorSize; i++) {
        auto minValue = std::min_element(distMat.ptr<float>(i, 0), distMat.ptr<float>(i, 0+distMat.cols));
        if (threshold_ == -1 || *minValue < threshold_) {
            int minIndex = std::distance(distMat.ptr<float>(i, 0), minValue);
            auto objvec = srcObjectList.objectvec(i);
            MxpiObject* dstMxpiObject = dstMxpiObjectList.add_objectvec();
            MxpiMetaHeader* dstMxpiMetaHeaderList = dstMxpiObject->add_headervec();
            dstMxpiMetaHeaderList->set_datasource(objectSource_);
            dstMxpiMetaHeaderList->set_memberid(0);
            
            dstMxpiObject->set_x0(objvec.x0());
            dstMxpiObject->set_y0(objvec.y0());
            dstMxpiObject->set_x1(objvec.x1());
            dstMxpiObject->set_y1(objvec.y1());
            
            // Generate ClassList
            MxpiClass* dstMxpiClass = dstMxpiObject->add_classvec();
            MxpiMetaHeader* dstMxpiMetaHeaderList_c = dstMxpiClass->add_headervec();
            dstMxpiMetaHeaderList_c->set_datasource(objectSource_);
            dstMxpiMetaHeaderList_c->set_memberid(0);
            dstMxpiClass->set_classid(i);
            dstMxpiClass->set_confidence(objvec.classvec(0).confidence());
            dstMxpiClass->set_classname(galleryIds[minIndex]);
        }
    }
}

void PluginFeatureMatch::ReadGalleryFeatures(std::string featuresPath,
                                             std::string idsPath, Mat &galleryFeatures,
                                             std::vector<std::string> &galleryIds) {
    // 读取gallery的人名
    std::ifstream ifile(idsPath);
	std::string str1;
	while (std::getline(ifile, str1)) {
		galleryIds.push_back(str1);
	}
	ifile.close();

    // 读取gallery特征库
    int featureLen = 512;
    const char* feaPath = featuresPath.c_str();
    FILE* fp = fopen(feaPath, "rb");
    galleryFeatures = Mat::zeros(int(galleryIds.size()), featureLen, CV_32FC1);
    for (int i = 0; i < int(galleryIds.size()); i++)
    {
        for (int j = 0; j < featureLen; j++)
        {
            fread(&galleryFeatures.at<float>(i, j), 1, sizeof(float), fp);
        }
    }
    fclose(fp);
    fp = NULL;
}

APP_ERROR PluginFeatureMatch::Process(std::vector<MxpiBuffer *> &mxpiBuffer) {
    LogInfo << "Begin to process PluginFeatureMatch.";
    // Get MxpiClassList from MxpiBuffer
    MxpiBuffer *inputMxpiBuffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*inputMxpiBuffer);
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    MxpiBuffer *inputMxpiBuffer1 = mxpiBuffer[1];
    MxpiMetadataManager mxpiMetadataManager1(*inputMxpiBuffer1);
    auto errorInfoPtr1 = mxpiMetadataManager1.GetErrorInfo();
    MxpiErrorInfo mxpiErrorInfo;
    if (errorInfoPtr != nullptr | errorInfoPtr1 != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_)
                   << "PluginFeatureMatch process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        LogError << "PluginFeatureMatch process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }

    // check data source
    APP_ERROR ret = CheckDataSource(mxpiMetadataManager, mxpiMetadataManager1);
    if (ret != APP_ERR_OK) {
        SendData(0, *inputMxpiBuffer1);
        return ret;
    }
    std::shared_ptr<void> queryListMetadata = mxpiMetadataManager.GetMetadata(querySource_);
    std::shared_ptr<void> objectListMetadata = mxpiMetadataManager1.GetMetadata(objectSource_);
    if (queryListMetadata == nullptr || objectListMetadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "Metadata is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        return APP_ERR_METADATA_IS_NULL;
    }
    std::shared_ptr<MxpiTensorPackageList> srcQueryListPtr = std::static_pointer_cast<MxpiTensorPackageList>(queryListMetadata);
    std::shared_ptr<MxpiObjectList> srcObjectListPtr = std::static_pointer_cast<MxpiObjectList>(objectListMetadata);
    std::shared_ptr<MxpiObjectList> resultObjectListPtr = std::make_shared<MxpiObjectList>();
    int objectSize = (*srcObjectListPtr).objectvec_size();
    int tensorSize = (*srcQueryListPtr).tensorpackagevec_size();
    
    Mat distMat = Mat(tensorSize, int(galleryIds.size()), CV_32FC1);
    if (objectSize > 0 && objectSize == tensorSize) {
        ret = ComputeDistance(*srcQueryListPtr, galleryFeatures, tensorSize, distMat);
    }

    GenerateOutput(*srcObjectListPtr, distMat, tensorSize, galleryIds, *resultObjectListPtr);
    ret = mxpiMetadataManager1.AddProtoMetadata(pluginName_, std::static_pointer_cast<void>(resultObjectListPtr));
    if (ret != APP_ERR_OK) {
        LogError << ErrorInfo_.str();
        SendMxpiErrorInfo(*inputMxpiBuffer1, pluginName_, ret, ErrorInfo_.str());
        SendData(0, *inputMxpiBuffer1);
    }
    // Send the data to downstream plugin
    SendData(0, *inputMxpiBuffer1);

    LogInfo << "End to process PluginFeatureMatch(" << elementName_ << ").";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> PluginFeatureMatch::DefineProperties() {
    std::vector<std::shared_ptr<void>> properties;
    // Get the action category from previous plugin
    auto querySource = std::make_shared<ElementProperty<std::string>>(ElementProperty<std::string> {
            STRING,
            "querySource",
            "queryFeatureSource",
            "query infer output ",
            "default", "NULL", "NULL"
    });

    auto objectSource = std::make_shared<ElementProperty<std::string>>(ElementProperty<std::string> {
            STRING,
            "objectSource",
            "queryobjectSource",
            "detection infer postprocess output ",
            "default", "NULL", "NULL"
    });

    // The action of interest file path
    auto galleryFeaturesPath = std::make_shared<ElementProperty<std::string>>(ElementProperty<std::string> {
            STRING,
            "galleryFeaturesPath",
            "features of gallery file path",
            "the path of gallery images features file",
            "NULL", "NULL", "NULL"
    });

    auto galleryIdsPath = std::make_shared<ElementProperty<std::string>>(ElementProperty<std::string> {
            STRING,
            "galleryIdsPath",
            "id of gallery file path",
            "the path of gallery person name file",
            "NULL", "NULL", "NULL"
    });

    auto metric = std::make_shared<ElementProperty<std::string>>(ElementProperty<std::string> {
            STRING,
            "metric",
            "metric",
            "the method of compute distance, select from 'euclidean', 'cosine'.",
            "euclidean", "NULL", "NULL"
    });

    auto threshold = std::make_shared<ElementProperty<float>>(ElementProperty<float> {
            FLOAT,
            "threshold",
            "distanceThreshold",
            "if distance is more than threshold,not matched gallery",
            -1.0, -1.0, 1000.0
    });

    properties.push_back(querySource);
    properties.push_back(objectSource);
    properties.push_back(galleryFeaturesPath);
    properties.push_back(galleryIdsPath);
    properties.push_back(metric);
    properties.push_back(threshold);
    return properties;
}

MxpiPortInfo PluginFeatureMatch::DefineInputPorts() {
    MxpiPortInfo inputPortInfo;
    std::vector<std::vector<std::string>> value = {{"ANY"}, {"ANY"}};
    GenerateStaticInputPortsInfo(value, inputPortInfo);
    return inputPortInfo;
}

MxpiPortInfo PluginFeatureMatch::DefineOutputPorts() {
    MxpiPortInfo outputPortInfo;
    // Output: {{MxpiObjectList}}
    std::vector<std::vector<std::string>> value = {{"ANY"}};
    GenerateStaticOutputPortsInfo(value, outputPortInfo);
    return outputPortInfo;
}

namespace {
    MX_PLUGIN_GENERATE(PluginFeatureMatch)
}
