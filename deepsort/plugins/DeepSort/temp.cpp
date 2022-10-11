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
APP_ERROR MxpiMotSimpleSortBase::GetModelInferResult(MxpiBuffer &buffer, std::vector<DetectObject> &detectObjectList)
{
    MxpiMetadataManager mxpiMetadataManager(buffer);
    APP_ERROR ret = CheckDataStructure(mxpiMetadataManager);
    if (ret != APP_ERR_OK) {
        LogError << errorInfo_.str();
        return ret;
    }
    std::shared_ptr<MxpiObjectList> objectList = std::static_pointer_cast<MxpiObjectList>(
        mxpiMetadataManager.GetMetadata(dataSourceDetection_));
    std::shared_ptr<MxpiFeatureVectorList> featureList;
    if (objectList->objectvec_size() == 0) {
        LogDebug << "Object detection result of model infer is null.";
        return APP_ERR_OK;
    }
    if (withFeature_) {
        featureList = std::static_pointer_cast<MxpiFeatureVectorList>(
            mxpiMetadataManager.GetMetadata(dataSourceFeature_));
        if (featureList->featurevec_size() == 0) {
            ret = APP_ERR_COMM_FAILURE;
            errorInfo_ << GetError(ret, elementName_) << "Face short feature result of model infer is null.";
            return ret;
        }
    }
    for (int i = 0; i < objectList->objectvec_size(); ++i) {
        DetectObject detectObject {};
        detectObject.detectInfo = objectList->objectvec(i);
        detectObject.memberId = static_cast<uint32_t>(i);
        if (withFeature_) {
            GetFeatureVector(featureList, i, detectObject);
        }
        detectObjectList.push_back(detectObject);
    }
    return APP_ERR_OK;
}

void MxpiMotSimpleSortBase::GetFeatureVector(const std::shared_ptr<MxTools::MxpiFeatureVectorList> &featureList,
    const int32_t &memberId, DetectObject &detectObject)
{
    for (int i = 0; i < featureList->featurevec_size(); ++i) {
        if (featureList->featurevec(i).headervec_size() == 0) {
            LogError << GetError(APP_ERR_COMM_OUT_OF_RANGE) << "protobuf message vector is invalid.";
            return;
        }
        if (featureList->featurevec(i).headervec(0).memberid() == memberId) {
            detectObject.featureVector = featureList->featurevec(i);
        }
    }
}
