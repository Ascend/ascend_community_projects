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
#include "FairmotPostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"
#include "MxTools/Proto/MxpiDataType.pb.h"
#include "MxTools/PluginToolkit/buffer/MxpiBufferManager.h"
#include <typeinfo>
#define TWO 2
#define THREE 3
#define FOUR 4
#define REDUCE 0.5
#define REDUCE_NEG (-0.5)
using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
namespace {
    const string INPUT_SHAPE_TYPE = "MxpiTensorPackageList";
    const string METADATA_KEY_OBJ = "mxpi_fairmot_obj";
    const string METADATA_KEY_FEA = "mxpi_fairmot_fea";
    const float CONF_THRES = 0.35;
    auto uint8Deleter = [] (uint8_t* p) { };
}

void GetTensors(const MxTools::MxpiTensorPackageList tensorPackageList,
                std::vector<MxBase::TensorBase> &tensors) {
    for (int i = 0; i < tensorPackageList.tensorpackagevec_size(); ++i) {
        for (int j = 0; j < tensorPackageList.tensorpackagevec(i).tensorvec_size(); j++) {
            MxBase::MemoryData memoryData = {};
            memoryData.deviceId = tensorPackageList.tensorpackagevec(i).tensorvec(j).deviceid();
            memoryData.type = (MxBase::MemoryData::MemoryType)tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).memtype();
            memoryData.size = (uint32_t) tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).tensordatasize();
            memoryData.ptrData = (void *) tensorPackageList.
                    tensorpackagevec(i).tensorvec(j).tensordataptr();
            std::vector<uint32_t> outputShape = {};
            for (int k = 0; k < tensorPackageList.
            tensorpackagevec(i).tensorvec(j).tensorshape_size(); ++k) {
                outputShape.push_back((uint32_t) tensorPackageList.
                tensorpackagevec(i).tensorvec(j).tensorshape(k));
            }
            MxBase::TensorBase tmpTensor(memoryData, true, outputShape,
                                         (MxBase::TensorDataType)tensorPackageList.
                                         tensorpackagevec(i).tensorvec(j).tensordatatype());
            tensors.push_back(tmpTensor);
        }
    }
}
void FairmotPostProcess::CoordinatesReduction(const uint32_t index,
                                              const ResizedImageInfo &resizedImageInfo,
                                              vector<ObjectInfo> &objInfos,
                                              bool normalizedFlag)
{
    if (!normalizedFlag) {
        LogError << "Error CoordinatesReduction type in this example.";
    }
    int imgWidth = resizedImageInfo.widthOriginal;
    int imgHeight = resizedImageInfo.heightOriginal;
    float ratio = resizedImageInfo.keepAspectRatioScaling;
    for (auto objInfo = objInfos.begin(); objInfo != objInfos.end();) {
        objInfo->x0 *= resizedImageInfo.widthResize / ratio;
        objInfo->y0 *= resizedImageInfo.heightResize / ratio;
        objInfo->x1 *= resizedImageInfo.widthResize / ratio;
        objInfo->y1 *= resizedImageInfo.heightResize / ratio;
        if (objInfo->x0 > imgWidth || objInfo->y0 > imgHeight) {
            objInfo = objInfos.erase(objInfo);
            continue;
        }
        if (objInfo->x1 > imgWidth) {
            objInfo->x1 = imgWidth;
        }
        if (objInfo->y1 > imgHeight) {
            objInfo->y1 = imgHeight;
        }
        ++objInfo;
    }
}
APP_ERROR FairmotPostProcess::Init(std::map<std::string, std::shared_ptr<void>>& configParamMap) {
    LogInfo << "FairmotPostProcess::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    std::shared_ptr<string> descriptionMessageProSptr = std::static_pointer_cast<string>(configParamMap["descriptionMessage"]);
    descriptionMessage_ = *descriptionMessageProSptr.get();
    return APP_ERR_OK;
}

APP_ERROR FairmotPostProcess::DeInit() {
    LogInfo << "FairmotPostProcess::DeInit end.";
    return APP_ERR_OK;
}

bool FairmotPostProcess::IsValidTensors(const std::vector <TensorBase> &tensors) {
    int fairmotType_ = 4;
    if (tensors.size() != (size_t) fairmotType_) {
        LogError << "number of tensors (" << tensors.size() << ") " << "is unequal to fairmotType_("
                << fairmotType_ << ")";
        return false;
    }
    return true;
}

APP_ERROR FairmotPostProcess::SetMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
    const MxpiErrorInfo mxpiErrorInfo) {
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

APP_ERROR FairmotPostProcess::PrintMxpiErrorInfo(MxpiBuffer& buffer, const std::string pluginName,
    MxpiErrorInfo mxpiErrorInfo, APP_ERROR app_error, std::string errorName)
{
    ErrorInfo_ << GetError(app_error, pluginName_) << errorName;
    LogError << errorName;
    mxpiErrorInfo.ret = app_error;
    mxpiErrorInfo.errorInfo = ErrorInfo_.str();
    SetMxpiErrorInfo(buffer, pluginName_, mxpiErrorInfo);
    return app_error;
}

APP_ERROR FairmotPostProcess::GenerateresizedImageInfos(vector<MxpiBuffer*> mxpiBuffer,
                                                        const MxpiTensorPackageList srcMxpiTensorPackage,
                                                        vector <ResizedImageInfo> &resizedImageInfos)
{
    auto dataSourceResize = srcMxpiTensorPackage.tensorpackagevec(0).headervec(0).datasource();
    MxTools::MxpiMetadataManager mxpiMxpiMetadataManager(*mxpiBuffer[0]);
    auto mxpiVisionList = std::static_pointer_cast<MxTools::MxpiVisionList>(
        mxpiMxpiMetadataManager.GetMetadataWithType(dataSourceResize, "MxpiVisionList"));
    if (mxpiVisionList == nullptr)
    {
        LogWarn << "Get mxpiVisionList failed from " << dataSourceResize;
        return APP_ERR_OK;
    }
    else
    {
        for (size_t i = 0; i < mxpiVisionList->visionvec_size(); i++)
        {
            auto info = mxpiVisionList->visionvec(i).visioninfo();
            MxBase::ResizedImageInfo reseizedImageInfo {
                info.width(), info.height(), 0, 0, (MxBase::ResizeType)info.resizetype(), info.keepaspectratioscaling()
            };
            MxTools::MxpiFrame frameData = MxpiBufferManager::GetDeviceDataInfo(*mxpiBuffer[0]);
            if (frameData.visionlist().visionvec_size() == 0)
            {
                reseizedImageInfo.widthOriginal = info.width();
                reseizedImageInfo.heightOriginal = info.height();
            }
            else
            {
                reseizedImageInfo.widthOriginal = frameData.visionlist().visionvec(0).visioninfo().width();
                reseizedImageInfo.heightOriginal = frameData.visionlist().visionvec(0).visioninfo().height();
            }
            resizedImageInfos.push_back(reseizedImageInfo);
            }
        }
    return APP_ERR_OK;
}

int FairmotPostProcess::ObjectDetectionOutput(
    const vector <TensorBase> &tensors,
    vector <vector<ObjectInfo>> &objectInfos,
    vector<vector<float>> &ID_feature,
    const vector <ResizedImageInfo> &resizedImageInfos)
{
    LogDebug << "FairmotPostProcess start to write results.";
    if (tensors.size() == 0) {
        return 0;
    }
    auto shape = tensors[0].GetShape();
    if (shape.size() == 0) {
        return 0;
    }
    std::vector <std::shared_ptr<void>> featLayerData = {};
    std::vector <std::vector<size_t>> featLayerShapes = {};
    for (uint32_t j = 0; j < tensors.size(); j++) {
        auto dataPtr = (uint8_t *)tensors[j].GetBuffer();
        std::shared_ptr<void> tmpPointer;
        tmpPointer.reset(dataPtr, uint8Deleter);

        featLayerData.push_back(tmpPointer);
        shape = tensors[j].GetShape();

        std::vector <size_t> featLayerShape = {};
        for (auto s : shape) {
            featLayerShape.push_back((size_t) s);
        }
        featLayerShapes.push_back(featLayerShape);
    }

    std::shared_ptr<void> hm_addr = featLayerData[0];
    std::vector<std::vector<int>> XY;
    for (uint32_t i = 0; i < featLayerShapes[0][1] * featLayerShapes[0][TWO]; i++) {
        if (static_cast<float *>(hm_addr.get())[i] > CONF_THRES)
        {
            std::vector<int>xy;
            int x = i / featLayerShapes[0][2];
            int y = i - featLayerShapes[0][2] * x;
            xy.push_back(x);
            xy.push_back(y);
            XY.push_back(xy);
        }
    }
    if (XY.size() == 0) {
        return 1;
    }
    std::vector<float>scores;
    for (uint32_t i = 0; i < XY.size(); i++) {
        scores.push_back(static_cast<float *>(hm_addr.get())[XY[i][0] * featLayerShapes[0][TWO] + XY[i][1]]);
    }
    std::shared_ptr<void> wh_addr = featLayerData[1];
    std::shared_ptr<void> reg_addr = featLayerData[2];

    std::vector<std::vector<float>>WH;
    for (int i = 0; i < XY.size();i++) {
        std::vector<float>wh;
        for (int j = 0; j < featLayerShapes[1][THREE]; j++) {
            wh.push_back(static_cast<float *>(wh_addr.get())[(XY[i][0] * featLayerShapes[0][TWO] + XY[i][1]) * featLayerShapes[1][THREE] + j]);
        }
        WH.push_back(wh);
    }

    std::vector<std::vector<float>>REG;
    for (int i = 0; i < XY.size();i++) {
        std::vector<float>reg;
        for (int j = 0; j < featLayerShapes[TWO][THREE]; j++) {
            reg.push_back(static_cast<float *>(reg_addr.get())[(XY[i][0] * featLayerShapes[0][TWO] + XY[i][1]) * featLayerShapes[TWO][THREE] + j]);
        }
        REG.push_back(reg);
    }
    std::shared_ptr<void> id_addr = featLayerData[3];
    for (int i = 0; i < XY.size();i++) {
        std::vector<float>id_feature;
        for (int j = 0; j < featLayerShapes[THREE][THREE]; j++) {
            id_feature.push_back(static_cast<float *>(id_addr.get())[(XY[i][0] * featLayerShapes[0][TWO] + XY[i][1]) * featLayerShapes[THREE][THREE] + j]);
        }
        ID_feature.push_back(id_feature);
    }
    std::vector<std::vector<float>> XY_f;
    for (int i = 0; i < XY.size(); i++) {
        std::vector<float>xy_f;
        xy_f.push_back(XY[i][0]);
        xy_f.push_back(XY[i][1]);
        XY_f.push_back(xy_f);
    }

    for (int i = 0; i < XY_f.size(); i++) {
        XY_f[i][1] = XY_f[i][1] + REG[i][0];
        XY_f[i][0] = XY_f[i][0] + REG[i][1];
    }
    std::vector<std::vector<float>>dets;
    for (int i = 0; i < XY.size(); i++) {
        std::vector<float>det;
        det.push_back(XY_f[i][1] - WH[i][0]);
        det.push_back(XY_f[i][0] - WH[i][1]);
        det.push_back(XY_f[i][1] + WH[i][TWO]);
        det.push_back(XY_f[i][0] + WH[i][THREE]);
        det.push_back(scores[i]);
        det.push_back(0);
        dets.push_back(det);
    }

    int width = resizedImageInfos[0].widthOriginal;
    int height = resizedImageInfos[0].heightOriginal;
    int inp_height = resizedImageInfos[0].heightResize;
    int inp_width = resizedImageInfos[0].widthResize;

    std::vector<float>c;
    int half = 2;
    c.push_back(width / half);
    c.push_back(height / half);
    std::vector<float>center(c);
    float scale = 0;
    scale = std::max(float(inp_width) / float(inp_height) * height, float(width)) * 1.0 ;
    std::vector<float>Scale;
    Scale.push_back(scale);
    Scale.push_back(scale);
    int down_ratio = 4;
    int h = inp_height / down_ratio ;
    int w = inp_width / down_ratio ;
    std::vector<int>output_size;
    output_size.push_back(w);
    output_size.push_back(h);

    int rot = 0;
    std::vector<float>shift(TWO, 0);
    int inv = 1;

    std::vector<float>scale_tmp(Scale);
    float src_w = scale_tmp[0];
    int dst_w = output_size[0];
    int dst_h = output_size[1];

    float pi = 3.1415926;
    int dir = 180;
    float rot_rad = pi * rot / dir;

    std::vector<float>src_point;
    src_point.push_back(0);
    src_point.push_back(src_w * (REDUCE_NEG));

    float sn = sin(rot_rad);
    float cs = cos(rot_rad);

    std::vector<float>src_dir(TWO, 0);
    src_dir[0] = src_point[0] * cs - src_point[1] * sn ;
    src_dir[1] = src_point[0] * sn + src_point[1] * cs ;
    std::vector<float>dst_dir;
    dst_dir.push_back(0);
    dst_dir.push_back(dst_w * (REDUCE_NEG));
    float src[3][2] = {0};
    float dst[3][2] = {0};
    src[0][0] = center[0] + scale_tmp[0] * shift[0];
    src[0][1] = center[1] + scale_tmp[1] * shift[1];
    src[1][0] = center[0] + src_dir[0] + scale_tmp[0] * shift[0];
    src[1][1] = center[1] + src_dir[1] + scale_tmp[1] * shift[1];
    dst[0][0] = dst_w * REDUCE;
    dst[0][1] = dst_h * REDUCE;
    dst[1][0] = dst_w * REDUCE + dst_dir[0];
    dst[1][1] = dst_h * REDUCE + dst_dir[1];
    std::vector<float>direct;
    direct.push_back(src[0][0]-src[1][0]);
    direct.push_back(src[0][1]-src[1][1]);
    src[TWO][0] = src[1][0] - direct[1];
    src[TWO][1] = src[1][1] + direct[0];
    direct[0] = dst[0][0] - dst[1][0];
    direct[1] = dst[0][1] - dst[1][1];
    dst[TWO][0] = dst[1][0] - direct[1];
    dst[TWO][1] = dst[1][1] + direct[0];

    cv::Point2f SRC[3];
    cv::Point2f DST[3];
    SRC[0] = cv::Point2f(src[0][0], src[0][1]);
    SRC[1] = cv::Point2f(src[1][0], src[1][1]);
    SRC[TWO] = cv::Point2f(src[TWO][0], src[TWO][1]);
    DST[0] = cv::Point2f(dst[0][0], dst[0][1]);
    DST[1] = cv::Point2f(dst[1][0], dst[1][1]);
    DST[TWO] = cv::Point2f(dst[TWO][0], dst[TWO][1]);
    cv::Mat trans(TWO, THREE, CV_64FC1);
    if (inv == 1) {
        trans = cv::getAffineTransform(DST, SRC);
    }
    else {
        trans = cv::getAffineTransform(SRC, DST);
    }
 
    float Trans[2][3];
    for (int i = 0; i < TWO; i++) {
        for (int j = 0; j < THREE; j++) {
            Trans[i][j] = trans.at<double>(i, j);
        }
    }
    // affine_transform
    // Calculate the coordinates of bbox_top_left x, y
    for (int i = 0; i < dets.size(); i++) {
        float new_pt[3] = {dets[i][0], dets[i][1], 1 };
        dets[i][0] = Trans[0][0]* new_pt[0] + Trans[0][1]* new_pt[1] + Trans[0][TWO]* new_pt[TWO];
        dets[i][1] = Trans[1][0]* new_pt[0] + Trans[1][1]* new_pt[1] + Trans[1][TWO]* new_pt[TWO];
    }
    // Calculate the coordinates of bbox_bottom_right x, y
    for (int i = 0; i < dets.size(); i++) {
        float new_pt[3] = {dets[i][2], dets[i][3], 1 };
        dets[i][TWO] = Trans[0][0]* new_pt[0] + Trans[0][1]* new_pt[1] + Trans[0][TWO]* new_pt[TWO];
        dets[i][THREE] = Trans[1][0]* new_pt[0] + Trans[1][1]* new_pt[1] + Trans[1][TWO]* new_pt[TWO];
    }
    // output
    std::vector <ObjectInfo> objectInfo;
    for (int i = 0; i < dets.size(); i++) {
        ObjectInfo objInfo;
        objInfo.classId = 0;
        objInfo.confidence = dets[i][FOUR];
        objInfo.className = " ";
        // Normalization
        objInfo.x0 = dets[i][0] / resizedImageInfos[0].widthOriginal;
        objInfo.y0 = dets[i][1] / resizedImageInfos[0].heightOriginal;
        objInfo.x1 = dets[i][TWO] / resizedImageInfos[0].widthOriginal;
        objInfo.y1 = dets[i][THREE] / resizedImageInfos[0].heightOriginal;
        objectInfo.push_back(objInfo);
    }

    objectInfos.push_back(objectInfo);
    
    LogDebug << "FairmotPostProcess write results successed.";
    return TWO;
}
APP_ERROR FairmotPostProcess::GenerateOutput(const MxTools::MxpiTensorPackageList srcMxpiTensorPackage,
                                             std::vector <ResizedImageInfo> &resizedImageInfos,
                                             MxTools::MxpiObjectList& dstMxpiObjectList,
                                             MxpiFeatureVectorList& dstMxpiFeatureVectorList)
{
    // Get Tensor
    std::vector<MxBase::TensorBase> tensors = {};
    GetTensors(srcMxpiTensorPackage, tensors);
    // Check Tensor
    bool isValid = IsValidTensors(tensors);
    if (!isValid)
    {
        LogError << "Is unValid Tensors";
        return APP_ERR_ACL_OP_INPUT_NOT_MATCH;
    }
    // Compute objects
    std::vector<std::vector<ObjectInfo>> objectInfos;
    std::vector<std::vector<float>> ID_feature;
    int flag = ObjectDetectionOutput(tensors, objectInfos, ID_feature, resizedImageInfos);
    if (flag == 1) {
        // flag: 1 represents no pedestrians are detected
        MxpiObject* dstMxpiObject = dstMxpiObjectList.add_objectvec();
        MxpiClass* dstMxpiClass = dstMxpiObject->add_classvec();
        MxpiFeatureVector* mxpiFeature = dstMxpiFeatureVectorList.add_featurevec();
        MxpiMetaHeader* dstMxpiMetaHeaderList = mxpiFeature->add_headervec();
        return APP_ERR_OK;
    }
    else if (flag == TWO) {
        // flag: 2 represents pedestrian detected
        for (uint32_t i = 0; i < resizedImageInfos.size(); i++) {
            CoordinatesReduction(i, resizedImageInfos[i], objectInfos[i]);
        }
        // Generate ObjectList
        for (size_t i = 0; i < objectInfos[0].size(); i++)
        {
            auto objInfo = objectInfos[0][i];
            MxpiObject* dstMxpiObject = dstMxpiObjectList.add_objectvec();
            MxpiMetaHeader* dstMxpiMetaHeaderList = dstMxpiObject->add_headervec();
            dstMxpiMetaHeaderList->set_datasource(parentName_);
            dstMxpiMetaHeaderList->set_memberid(0);

            dstMxpiObject->set_x0(objInfo.x0);
            dstMxpiObject->set_y0(objInfo.y0);
            dstMxpiObject->set_x1(objInfo.x1);
            dstMxpiObject->set_y1(objInfo.y1);

            // Generate ClassList
            MxpiClass* dstMxpiClass = dstMxpiObject->add_classvec();
            MxpiMetaHeader* dstMxpiMetaHeaderList_c = dstMxpiClass->add_headervec();
            dstMxpiMetaHeaderList_c->set_datasource(parentName_);
            dstMxpiMetaHeaderList_c->set_memberid(0);
            dstMxpiClass->set_classid(objInfo.classId);
            dstMxpiClass->set_confidence(objInfo.confidence);
            dstMxpiClass->set_classname(objInfo.className);
        }
        
        for (size_t i = 0; i < ID_feature.size(); i++)
        {
            MxpiFeatureVector* mxpiFeature = dstMxpiFeatureVectorList.add_featurevec();
            MxpiMetaHeader* dstMxpiMetaHeaderList = mxpiFeature->add_headervec();
            dstMxpiMetaHeaderList->set_datasource(parentName_);
            dstMxpiMetaHeaderList->set_memberid(0);

            for (size_t j = 0; j < ID_feature[i].size(); j++)
            {
                mxpiFeature->add_featurevalues(ID_feature[i][j]);
            }
        }
        return APP_ERR_OK;
    }
    else {
        // flag: 0 represents the input from tensorinfer is empty
        LogError << "Is unValid Tensors";
        return APP_ERR_ACL_OP_INPUT_NOT_MATCH;
    }
}

APP_ERROR FairmotPostProcess::Process(std::vector<MxpiBuffer*>& mxpiBuffer) {
    LogInfo << "FairmotPostProcess::Process start";
    MxpiBuffer* buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, APP_ERR_COMM_FAILURE, "FairmotPostProcess process is not implemented");
    }
    // Get the data from buffer
    shared_ptr<void> metadata = mxpiMetadataManager.GetMetadata(parentName_);
    if (metadata == nullptr) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, APP_ERR_METADATA_IS_NULL, "Metadata is NULL, failed");
    }
    // check the proto struct name
    google::protobuf::Message* msg = (google::protobuf::Message*)metadata.get();
    const google::protobuf::Descriptor* desc = msg->GetDescriptor();
    if (desc->name() != INPUT_SHAPE_TYPE) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, APP_ERR_PROTOBUF_NAME_MISMATCH, "Proto struct name is not MxpiTensorPackageList, failed");
    }
    shared_ptr<MxpiTensorPackageList> srcMxpiTensorPackageListSptr = static_pointer_cast<MxpiTensorPackageList>(metadata);
    shared_ptr<MxpiObjectList> dstMxpiObjectList = make_shared<MxpiObjectList>();
    shared_ptr<MxpiFeatureVectorList> dstMxpiFeatureVectorList = make_shared<MxpiFeatureVectorList>();
    std::vector <ResizedImageInfo> resizedImageInfos;
    // Get resizedImageInfos
    APP_ERROR ret = GenerateresizedImageInfos(mxpiBuffer, *srcMxpiTensorPackageListSptr, resizedImageInfos);
    if (ret != APP_ERR_OK) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, ret, "Generate resizedImageInfos failed");
    }
    // Generate sample output
    ret = GenerateOutput(*srcMxpiTensorPackageListSptr, resizedImageInfos, *dstMxpiObjectList, *dstMxpiFeatureVectorList);
    if (ret != APP_ERR_OK) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, ret, "FairmotPostProcess gets inference information failed. Checkc tensor value!");
    }
    // Add Generated data to metedata
    ret = mxpiMetadataManager.AddProtoMetadata(METADATA_KEY_OBJ, static_pointer_cast<void>(dstMxpiObjectList));
    if (ret != APP_ERR_OK) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, ret, "FairmotPostProcess add MxpiObjectList metadata failed.");
    }
    ret = mxpiMetadataManager.AddProtoMetadata(METADATA_KEY_FEA, static_pointer_cast<void>(dstMxpiFeatureVectorList));
    if (ret != APP_ERR_OK) {
        return PrintMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo, ret, "FairmotPostProcess add MxpiFeatureVectorList metadata failed.");
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    LogInfo << "FairmotPostProcess::Process end";
    return APP_ERR_OK;
}

std::vector<std::shared_ptr<void>> FairmotPostProcess::DefineProperties()
{
    // Define an A to store properties
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parentNameProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "dataSource", "name", "the name of previous plugin", "mxpi_tensorinfer0", "NULL", "NULL"});
    auto descriptionMessageProSptr = std::make_shared<ElementProperty<string>>(ElementProperty<string> {
        STRING, "descriptionMessage", "message", "Description mesasge of plugin", "This is FairmotPostProcess", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    properties.push_back(descriptionMessageProSptr);
    return properties;
}

// Register the Sample plugin through macro
MX_PLUGIN_GENERATE(FairmotPostProcess)
