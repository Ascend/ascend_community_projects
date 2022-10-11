/*
 * Copyright (c) 2022.Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "DeepSort.h"
#include "MxBase/Log/Log.h"
#include "tracker.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iomanip>
#define TWO 2
#define THREE 3
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
const int nn_budget = 100;
static int frame_id = 0;
const float max_cosine_distance = 0.2;
int control = 900;
tracker mytracker(max_cosine_distance, nn_budget);

APP_ERROR DeepSort::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap)
{
    LogInfo << "DeepSort::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    std::shared_ptr<string> objectNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSourceDetection"]);
    objectName_ = *objectNamePropSptr.get();
    std::shared_ptr<string> featureNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSourceFeature"]);
    featureName_ = *featureNamePropSptr.get();
    return APP_ERR_OK;
}

APP_ERROR DeepSort::DeInit()
{
    LogInfo << "DeepSort::DeInit end.";
    return APP_ERR_OK;
}

APP_ERROR DeepSort::GenerateSampleOutput (const MxpiObjectList srcMxpiObjectList,
                                          const std::vector<TrackerInfo>& tracker_infos,
                                          MxpiTrackLetList& dstMxpiTrackLetList)
{
    for (int i = 0; i < tracker_infos.size(); i++) {
        const TrackerInfo& r = tracker_infos[i];
        int track_id  = r.trackId;
        int age       = r.age;
        int hits      = r.hits;
        int trackFlag = r.trackFlag;

        MxpiTrackLet* dstMxpiTrackLet         = dstMxpiTrackLetList.add_trackletvec();
        MxpiMetaHeader* dstMxpiMetaHeaderList = dstMxpiTrackLet->add_headervec();
        dstMxpiMetaHeaderList->set_datasource(parentName_);
        dstMxpiMetaHeaderList->set_memberid(i);
      
        dstMxpiTrackLet->set_trackid(track_id);
        dstMxpiTrackLet->set_age(age);
        dstMxpiTrackLet->set_hits(hits);
        dstMxpiTrackLet->set_trackflag(trackFlag);
    }
    return APP_ERR_OK;
}

APP_ERROR DeepSort::Process(
    std::vector<MxpiBuffer*>& mxpiBuffer) {
        LogInfo << "DeepSort::Process start";
        frame_id ++;
  
        MxpiBuffer* buffer = mxpiBuffer[0];
        MxpiMetadataManager mxpiMetadataManager(*buffer);
        MxpiErrorInfo mxpiErrorInfo;
        ErrorInfo_.str("");
        auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
        if (errorInfoPtr != nullptr) {
            ErrorInfo_ << GetError (APP_ERR_COMM_FAILURE, pluginName_) << "DeepSort process is not implemented";
            mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
            mxpiErrorInfo.errorInfo = ErrorInfo_.str();
            SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
            LogError << "DeepSort process is not implemented";
            return APP_ERR_COMM_FAILURE;
    }
 
        shared_ptr<void> objectMetadata = mxpiMetadataManager.GetMetadata(objectName_);
        if (objectMetadata == nullptr) {
            ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "objectMetadata is NULL, failed";
            mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
            mxpiErrorInfo.errorInfo = ErrorInfo_.str();
            SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
            return APP_ERR_METADATA_IS_NULL;
        }
  
    shared_ptr<void> featureMetadata = mxpiMetadataManager.GetMetadata(featureName_);
    if (featureMetadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "featureMetadata is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_METADATA_IS_NULL;
    }

    std::shared_ptr<MxpiObjectList> objectList = std::static_pointer_cast<MxpiObjectList>(
        mxpiMetadataManager.GetMetadata(objectName_));
    std::shared_ptr<MxpiFeatureVectorList> featureList;
    std::vector<DetectObject> detectObjectList;

    if (objectList->objectvec_size() == 0) {
        LogDebug << "Object detection result of model infer is null.";
        return APP_ERR_OK;
    }
    LogInfo << "object size : " << objectList->objectvec_size();

    featureList = std::static_pointer_cast<MxpiFeatureVectorList>(
        mxpiMetadataManager.GetMetadata(featureName_));
    if (featureList->featurevec_size() == 0) {
        APP_ERROR ret = APP_ERR_COMM_FAILURE;
        errorInfo_ << GetError(ret, featureName_) << "Face short feature result of model infer is null.";
        return ret;
    }
 
    for (int i = 0; i < objectList->objectvec_size(); ++i) {
        DetectObject detectObject {};
        detectObject.detectInfo = objectList->objectvec(i);
        detectObject.memberId = static_cast<uint32_t>(i);

        GetFeatureVector(featureList, i, detectObject);
        detectObjectList.push_back(detectObject);
    }

    DETECTIONS detections;
    for (int i = 0; i < detectObjectList.size(); ++i)
    {
        DETECTION_ROW detection;
        DetectObject& detectObject = detectObjectList[i];
        const MxTools::MxpiObject& detect_info = detectObject.detectInfo;
        float x0 = detect_info.x0();
        float y0 = detect_info.y0();
        float x1 = detect_info.x1();
        float y1 = detect_info.y1();
        detection. tlwh(0, 0) = y0;
        detection. tlwh(0, 1) = x0;
        detection. tlwh(0, TWO) = x1-x0;
        detection. tlwh(0, THREE) = y1-y0;
        const auto& class_info = detect_info.classvec();
        float confi = class_info[0].confidence();
        detection. confidence = confi;
        const auto& feature_Vector = detectObject.featureVector;
        for (int j = 0; j < feature_Vector.featurevalues_size(); j++)
        {
            float value = feature_Vector.featurevalues(j);
            detection. feature (0, j) = value;
        }
        detections.push_back(detection);
    }

    LogInfo << "predict ------";
    mytracker.predict();
    std::vector<std::pair<int, int>> det_track_idxs = mytracker.update(detections);
    sort (det_track_idxs.begin(), det_track_idxs.end(),
        [](std::pair<int, int>& d1, std::pair<int, int>& d2)
        {
            return d1.first < d2.first;
        });

    for (auto& track : mytracker.tracks)
    {
        LogInfo << "track id: " << track.track_id << ", age: " << track.age << ", hits: " << track.hits << ", state: " << track.state;
    }
    ofstream dataFile;
	  dataFile.open("gt.txt", ofstream::app);
	  fstream file("gt.txt", ios::app);
    std::vector<TrackerInfo> tracker_infos;
    for (const auto& det_track_idx : det_track_idxs)
    {
        int det_id = det_track_idx.first;
        int track_idx = det_track_idx.second;
        Track& track = mytracker.tracks[track_idx];
        LogInfo << track.track_id << ",";
   if (frame_id <= control) {
          dataFile<<frame_id<<","<< track.track_id<<","<<int(detections[det_id].tlwh(0, 1))<<","<<int(detections[det_id].tlwh(0, 0))<<","<<int(detections[det_id].tlwh(0, TWO))<<","<<int(detections[det_id].tlwh(0, THREE))<<","<<1<<","<< 1 <<","<< 1 <<","<<endl;
     }
          
        TrackerInfo tracker_info;
        tracker_info.trackId   = track.track_id;
        tracker_info.age       = track.age;
        tracker_info.hits      = track.hits;
        tracker_info.trackFlag = track.state == Track::Tentative ? 0 : track.state == Track::Confirmed ? 1 : track.state == Track::Deleted ? TWO : 0;
        tracker_infos.push_back(tracker_info);
    }
    shared_ptr<MxpiTrackLetList> dstMxpiTrackLetListSptr = make_shared<MxpiTrackLetList>();
    APP_ERROR ret = GenerateSampleOutput(*objectList, tracker_infos, *dstMxpiTrackLetListSptr);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret, pluginName_) << "DeepSort gets inference information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiTrackLetListSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "DeepSort add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    SendData(0, *buffer);
    LogInfo << "DeepSort::Process end";
    return APP_ERR_OK;
}

void DeepSort::GetFeatureVector(const std::shared_ptr<MxTools::MxpiFeatureVectorList> &featureList,
    const int32_t &memberId, DetectObject &detectObject)
{
    for (int i = 0; i < featureList->featurevec_size(); ++i) {
        if (featureList->featurevec(i).headervec_size() == 0) {
            LogError << GetError(APP_ERR_COMM_OUT_OF_RANGE) << "protobuf message vector is invalid.";
            return;
        }
        
        if (i == memberId) {
            detectObject.featureVector = featureList->featurevec(i);
        }
    }
}

APP_ERROR DeepSort::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
    const MxpiErrorInfo mxpiErrorInfo)
{
    APP_ERROR ret = APP_ERR_OK;
    
    MxpiMetadataManager mxpiMetadataManager(buffer);
    ret = mxpiMetadataManager.AddErrorInfo(pluginName, mxpiErrorInfo);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to AddErrorInfo.";
        return ret;
    }
    ret = SendData(0, buffer);
    return ret;
}

std::vector<std::shared_ptr<void>> DeepSort::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;
    
    auto objectNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "dataSourceDetection", "inputName", "the name of fairmotpostprocessor", "mxpi_fairmot_obj", "NULL", "NULL"});
    auto featureNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "dataSourceFeature", "inputName", "the name of fairmotpostprocessor", "mxpi_fairmot_fea", "NULL", "NULL"});
    properties.push_back(objectNameProSptr);
    properties.push_back(featureNameProSptr);
 
    return properties;
}

MX_PLUGIN_GENERATE(DeepSort)

