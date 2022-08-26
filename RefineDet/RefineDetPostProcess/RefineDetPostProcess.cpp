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

#include "RefineDetPostProcess.h"
#include "MxBase/Log/Log.h"
#include "MxBase/Maths/FastMath.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"

namespace {
    const int SCALE = 32;
    const int BIASESDIM = 2;
    const int OFFSETWIDTH = 2;
    const int OFFSETHEIGHT = 3;
    const int OFFSETBIASES = 1;
    const int OFFSETOBJECTNESS = 1;

    const int NHWC_HEIGHTINDEX = 1;
    const int NHWC_WIDTHINDEX = 2;
    const int NCHW_HEIGHTINDEX = 2;
    const int NCHW_WIDTHINDEX = 3;
    const int YOLO_INFO_DIM = 5;

    auto uint8Deleter = [](uint8_t *p) {};

    const uint32_t LEFTTOPX = 0;
    const uint32_t LEFTTOPY = 1;
    const uint32_t RIGHTTOPX = 2;
    const uint32_t RIGHTTOPY = 3;
    const int PRIOR_PARAMETERS[4] = {32, 64, 128, 256};
    const float IMAGE_WIDTH = 320.0;
    const float IMAGE_HEIGHT = 320.0;
    const float STEPS[4] = {8.0, 16.0, 32.0, 64.0};
    const float VARIANCE[2] = {0.1, 0.2};
    const uint32_t RECTANGLEPOINT = 4;
    const uint32_t DIM = 2; // image dimension is 2
    const uint32_t RECTANGLE_COLOR = 1;
    const uint32_t KEYPOINT_COLOR = 2;
    const uint32_t HINTPOINT_COLOR = 3;
    const uint32_t DIV_TWO = 2;
    const float TWO = 2.0;
}
using namespace MxBase;

RefineDetPostProcess &RefineDetPostProcess::operator=(const RefineDetPostProcess &other) {
    if (this == &other) {
        return *this;
    }
    ObjectPostProcessBase::operator=(other);
    objectnessThresh_ = other.objectnessThresh_; // Threshold of objectness value
    iouThresh_ = other.iouThresh_;
    anchorDim_ = other.anchorDim_;
    biasesNum_ = other.biasesNum_;
    yoloType_ = other.yoloType_;
    modelType_ = other.modelType_;
    yoloType_ = other.yoloType_;
    inputType_ = other.inputType_;
    biases_ = other.biases_;
    return *this;
}

APP_ERROR RefineDetPostProcess::Init(const std::map<std::string, std::shared_ptr<void>> &postConfig) {
    LogDebug << "Start to Init RefineDetPostProcess.";
    APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Fail to superInit in ObjectPostProcessBase.";
        return ret;
    }
    return APP_ERR_OK;

    configData_.GetFileValue<int>("BIASES_NUM", biasesNum_);
    std::string str;
    configData_.GetFileValue<std::string>("BIASES", str);
    configData_.GetFileValue<float>("OBJECTNESS_THRESH", objectnessThresh_);
    configData_.GetFileValue<float>("IOU_THRESH", iouThresh_);
    configData_.GetFileValue<int>("YOLO_TYPE", yoloType_);
    configData_.GetFileValue<int>("MODEL_TYPE", modelType_);
    configData_.GetFileValue<int>("YOLO_VERSION", yoloVersion_);
    configData_.GetFileValue<int>("INPUT_TYPE", inputType_);
    configData_.GetFileValue<int>("ANCHOR_DIM", anchorDim_);
    ret = GetBiases(str);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "Failed to get biases.";
        return ret;
    }
    LogDebug << "End to Init RefineDetPostProcess.";
    return APP_ERR_OK;
}

APP_ERROR RefineDetPostProcess::DeInit() {
    return APP_ERR_OK;
}

// 判断Tensor是否合法
bool RefineDetPostProcess::IsValidTensors(const std::vector<TensorBase> &tensors) const {
    if (tensors.size() != (size_t) yoloType_) {
        LogError << "number of tensors (" << tensors.size() << ") " << "is unequal to yoloType_("
                 << yoloType_ << ")";
        return false;
    }
    if (yoloVersion_ == RefineDet_VERSION) {
        for (size_t i = 0; i < tensors.size(); i++) {
            auto shape = tensors[i].GetShape();
            if (shape.size() < VECTOR_FIFTH_INDEX) {
                LogError << "dimensions of tensor [" << i << "] is less than " << VECTOR_FIFTH_INDEX << ".";
                return false;
            }
            uint32_t channelNumber = 1;
            int startIndex = modelType_ ? VECTOR_SECOND_INDEX : VECTOR_FOURTH_INDEX;
            int endIndex = modelType_ ? (shape.size() - VECTOR_THIRD_INDEX) : shape.size();
            for (int i = startIndex; i < endIndex; i++) {
                channelNumber *= shape[i];
            }
            if (channelNumber != anchorDim_ * (classNum_ + YOLO_INFO_DIM)) {
                LogError << "channelNumber(" << channelNumber << ") != anchorDim_ * (classNum_ + 5).";
                return false;
            }
        }
        return true;
    } else {
        return true;
    }
}

void RefineDetPostProcess::generate_objectInfos(const std::vector <TensorBase>& tensors,
                                                std::vector <std::vector<ObjectInfo>>& objectInfos,
                                                const std::vector <ResizedImageInfo>& resizedImageInfos,
                                                cv::Mat& res)
    {
        auto asm_conf = tensors[1].GetBuffer();
        auto odm_conf = tensors[3].GetBuffer();
        auto shape = tensors[0].GetShape();
        float width_resize_scale = (float)resizedImageInfos[0].widthResize / resizedImageInfos[0].widthOriginal;
        float height_resize_scale = (float)resizedImageInfos[0].heightResize / resizedImageInfos[0].heightOriginal;
        uint32_t batchSize = shape[0];
        uint32_t VectorNum = shape[1];

        for (uint32_t i = 0; i < batchSize; i++)
        {
            std::vector <ObjectInfo> objectInfo;
            auto asm_dataPtr_Conf = (float *) asm_conf + i * tensors[1].GetByteSize() / batchSize;
            auto odm_dataPtr_Conf = (float *) odm_conf + i * tensors[3].GetByteSize() / batchSize;

            for (uint32_t j = 0; j < VectorNum; j++) {
                float* asm_begin_Conf = asm_dataPtr_Conf + j * 2;
                float* odm_begin_Conf = odm_dataPtr_Conf + j * classNum_;

                for (uint32_t k = 1; k < classNum_; k++)
                {
                    float conf = *(asm_begin_Conf) <= 0.01 ? 0 : *(odm_begin_Conf + k);
                    if (conf > confidenceThresh_)
                    {
                        ObjectInfo objInfo;
                        objInfo.confidence = conf;
                        objInfo.classId = k;
                        objInfo.className = configData_.GetClassName(k);
                        objInfo.x0 = res.at<float>(j, LEFTTOPX) * IMAGE_WIDTH / width_resize_scale;
                        objInfo.y0 = res.at<float>(j, LEFTTOPY) * IMAGE_HEIGHT / height_resize_scale;
                        objInfo.x1 = res.at<float>(j, RIGHTTOPX) * IMAGE_WIDTH / width_resize_scale;
                        objInfo.y1 = res.at<float>(j, RIGHTTOPY) * IMAGE_HEIGHT / height_resize_scale;

                        objInfo.x0 = objInfo.x0 > 1 ? objInfo.x0 : 1;
                        objInfo.y0 = objInfo.y0 > 1 ? objInfo.y0 : 1;
                        objInfo.x1 = objInfo.x1 < resizedImageInfos[0].widthOriginal ? objInfo.x1 : resizedImageInfos[0].widthOriginal;
                        objInfo.y1 = objInfo.y1 < resizedImageInfos[0].heightOriginal ? objInfo.y1 : resizedImageInfos[0].heightOriginal;
                        objectInfo.push_back(objInfo);
                    }
                }
            }
            MxBase::NmsSort(objectInfo, iouThresh_);
            objectInfos.push_back(objectInfo);
        }
    }

// 将处理好的推理结果放入ObjectInfo
void RefineDetPostProcess::ObjectDetectionOutput(const std::vector<TensorBase> &tensors,
                                                 std::vector<std::vector<ObjectInfo>> &objectInfos,
                                                 const std::vector<ResizedImageInfo> &resizedImageInfos) {
    LogDebug << "RefineDetPostProcess start to write results.";
    if (tensors.size() == 0) {
        return;
    }

    auto asm_loc = tensors[0].GetBuffer();
    auto odm_loc = tensors[2].GetBuffer();
    auto shape = tensors[2].GetShape();

    cv::Mat PriorBox;
    cv::Mat asm_location = cv::Mat(shape[1], shape[2], CV_32FC1, asm_loc);
    cv::Mat odm_location = cv::Mat(shape[1], shape[2], CV_32FC1, odm_loc);
    GeneratePriorBox(PriorBox);
    cv::Mat res = decode_for_loc(asm_location, PriorBox);
    res = center_size(res);
    res = decode_for_loc(odm_location, res);

    generate_objectInfos(tensors, objectInfos, resizedImageInfos, res);

    LogDebug << "RefineDetPostProcess write results successed.";
}

APP_ERROR RefineDetPostProcess::Process(const std::vector<TensorBase> &tensors,
                                        std::vector<std::vector<ObjectInfo>> &objectInfos,
                                        const std::vector<ResizedImageInfo> &resizedImageInfos,
                                        const std::map<std::string, std::shared_ptr<void>> &paramMap) {
    LogDebug << "Start to Process RefineDetPostProcess.";
    APP_ERROR ret = APP_ERR_OK;
    if (resizedImageInfos.size() == 0) {
        ret = APP_ERR_INPUT_NOT_MATCH;
        LogError << GetError(ret) << "resizedImageInfos is not provided which is necessary for RefineDetPostProcess.";
        return ret;
    }
    auto inputs = tensors;
    ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
        LogError << GetError(ret) << "CheckAndMoveTensors failed.";
        return ret;
    }

    ObjectDetectionOutput(inputs, objectInfos, resizedImageInfos);
    LogObjectInfos(objectInfos);

    LogDebug << "End to Process RefineDetPostProcess.";
    return APP_ERR_OK;
}

void RefineDetPostProcess::GeneratePriorBox(cv::Mat &anchors)
    {
        std::vector<std::vector<int>>feature_maps(RECTANGLEPOINT, std::vector<int>(DIM));
        for (uint32_t i = 0; i < feature_maps.size(); i++) {
            feature_maps[i][0] = IMAGE_HEIGHT / STEPS[i];
            feature_maps[i][1] = IMAGE_WIDTH / STEPS[i];
        }
        for (uint32_t k = 0; k < feature_maps.size(); k++) {
            auto f = feature_maps[k];
            float step = (float)STEPS[k];
            for (int i = 0; i < f[0]; i++) {
                for (int j = 0; j < f[1]; j++) {
                    int min_size = PRIOR_PARAMETERS[k];
                    cv::Mat anchor(1, RECTANGLEPOINT, CV_32F);
                    float center_x = (j + 0.5f) * step / IMAGE_WIDTH;
                    float center_y = (i + 0.5f) * step / IMAGE_HEIGHT;
                    float step_x = min_size / IMAGE_WIDTH;
                    float step_y = min_size / IMAGE_HEIGHT;

                    anchor.at<float>(0, LEFTTOPX) = center_x;
                    anchor.at<float>(0, LEFTTOPY) = center_y;
                    anchor.at<float>(0, RIGHTTOPX) = step_x;
                    anchor.at<float>(0, RIGHTTOPY) = step_y;
                    anchors.push_back(anchor);

                    anchor.at<float>(0, LEFTTOPX) = center_x;
                    anchor.at<float>(0, LEFTTOPY) = center_y;
                    anchor.at<float>(0, RIGHTTOPX) = step_x * sqrtf(TWO);
                    anchor.at<float>(0, RIGHTTOPY) = step_y / sqrtf(TWO);
                    anchors.push_back(anchor);

                    anchor.at<float>(0, LEFTTOPX) = center_x;
                    anchor.at<float>(0, LEFTTOPY) = center_y;
                    anchor.at<float>(0, RIGHTTOPX) = step_x / sqrtf(TWO);
                    anchor.at<float>(0, RIGHTTOPY) = step_y * sqrtf(TWO);
                    anchors.push_back(anchor);
                }
            }
        }
    }
    /*
     * @description: Generate prior boxes for detection boxes decoding
     * @param loc:  The matrix which contains box biases, shape[21824, 4]
     * @param prior: The matrix which contains prior box coordinates, shape[21824,4]
     * @param resize_scale_factor: The factor of min(WidthOriginal/WidthResize, HeightOriginal/HeightResize)
     * @param boxes: The matrix which contains detection box coordinates(x0,y0,x1,y1), shape[21824,4]
     */

    cv::Mat RefineDetPostProcess::center_size(cv::Mat boxes)
    {
        cv::Mat boxes_first = boxes.colRange(0, 2);
        cv::Mat boxes_last = boxes.colRange(2, 4);
        cv::Mat ret_boxes;
        cv::hconcat((boxes_first + boxes_last) / DIV_TWO, boxes_last - boxes_first, ret_boxes);
        return ret_boxes;
    }


    cv::Mat RefineDetPostProcess::decode_for_loc(cv::Mat &loc, cv::Mat &prior) {
        cv::Mat loc_first = loc.colRange(0, 2);
        cv::Mat loc_last = loc.colRange(2, 4);
        cv::Mat prior_first = prior.colRange(0, 2);
        cv::Mat prior_last = prior.colRange(2, 4);

        cv::Mat boxes1 = prior_first + (loc_first * VARIANCE[0]).mul(prior_last);
        cv::Mat boxes2;
        cv::exp(loc_last * VARIANCE[1], boxes2);
        boxes2 = boxes2.mul(prior_last);
        boxes1 = boxes1 - boxes2 / DIV_TWO;
        boxes2 = boxes2 + boxes1;

        cv::Mat boxes;
        cv::hconcat(boxes1, boxes2, boxes);
        return boxes;
    }

/*
 * @description: Compare the confidences between 2 classes and get the larger one
 * @描述：比较两个类之间的置信度，得到较大的一个
 */
void RefineDetPostProcess::CompareProb(int &classID, float &maxProb, float classProb, int classNum) {
    if (classProb > maxProb) {
        maxProb = classProb;
        classID = classNum;
    }
}

void RefineDetPostProcess::CodeDuplicationSetDet(float &x, float &y, float &width, float &height) {
    MxBase::ObjectInfo det;
    det.x0 = std::max(0.0f, x - width / COORDINATE_PARAM);
    det.x1 = std::min(1.0f, x + width / COORDINATE_PARAM);
    det.y0 = std::max(0.0f, y - height / COORDINATE_PARAM);
    det.y1 = std::min(1.0f, y + height / COORDINATE_PARAM);
}

void RefineDetPostProcess::SelectClassNCHW(std::shared_ptr<void> netout, NetInfo info,
                                           std::vector<MxBase::ObjectInfo> &detBoxes, int stride, OutputLayer layer) {
    for (int j = 0; j < stride; ++j) {
        for (int k = 0; k < info.anchorDim; ++k) {
            int bIdx = (info.bboxDim + 1 + info.classNum) * stride * k + j;
            int oIdx = bIdx + info.bboxDim * stride;

            float objectness = fastmath::sigmoid(static_cast<float *>(netout.get())[oIdx]);
            if (objectness <= objectnessThresh_) {
                continue;
            }
            int classID = -1;
            float maxProb = scoreThresh_;
            float classProb = 0.0;

            for (int c = 0; c < info.classNum; ++c) {
                classProb = fastmath::sigmoid(static_cast<float *>
                        (netout.get())[bIdx + (info.bboxDim + OFFSETOBJECTNESS + c) * stride]) * objectness;
                CompareProb(classID, maxProb, classProb, c);
            }
            if (classID < 0) {continue;}

            MxBase::ObjectInfo det;
            int row = j / layer.width;
            int col = j % layer.width;
            float x = (col + fastmath::sigmoid(static_cast<float *>(netout.get())[bIdx])) / layer.width;
            float y = (row + fastmath::sigmoid(static_cast<float *>(netout.get())[bIdx + stride])) / layer.height;
            float width = fastmath::exp(static_cast<float *>(netout.get())[bIdx + OFFSETWIDTH * stride]) *
                          layer.anchors[BIASESDIM * k] / info.netWidth;
            float height = fastmath::exp(static_cast<float *>(netout.get())[bIdx + OFFSETHEIGHT * stride]) *
                           layer.anchors[BIASESDIM * k + OFFSETBIASES] / info.netHeight;

            CodeDuplicationSetDet(x, y, width, height);
            det.classId = classID;
            det.className = configData_.GetClassName(classID);
            det.confidence = maxProb;
            if (det.confidence < separateScoreThresh_[classID]) {
                continue;
            }
            detBoxes.emplace_back(det);
        }
    }
}

void RefineDetPostProcess::SelectClassNCHWC(std::shared_ptr<void> netout, NetInfo info,
                                            std::vector<MxBase::ObjectInfo> &detBoxes, int stride, OutputLayer layer) {
    LogDebug << " out size " << sizeof(netout.get());
    const int offsetY = 1;
    for (int j = 0; j < stride; ++j) {
        for (int k = 0; k < info.anchorDim; ++k) {
            int bIdx = (info.bboxDim + 1 + info.classNum) * stride * k +
                       j * (info.bboxDim + 1 + info.classNum);
            int oIdx = bIdx + info.bboxDim;
            float objectness = fastmath::sigmoid(static_cast<float *>(netout.get())[oIdx]);
            if (objectness <= objectnessThresh_) {continue;}
            int classID = -1;
            float maxProb = scoreThresh_;
            float classProb = 0.0;

            for (int c = 0; c < info.classNum; ++c) {
                classProb = fastmath::sigmoid(static_cast<float *>(netout.get())[bIdx +
                                                                                 (info.bboxDim + OFFSETOBJECTNESS +
                                                                                  c)]) * objectness;
                CompareProb(classID, maxProb, classProb, c);
            }
            if (classID < 0) continue;
            MxBase::ObjectInfo det;
            int row = j / layer.width;
            int col = j % layer.width;
            float x = (col + fastmath::sigmoid(static_cast<float *>(netout.get())[bIdx]) * COORDINATE_PARAM -
                       MEAN_PARAM) / layer.width;
            float y = (row + fastmath::sigmoid(static_cast<float *>(netout.get())[bIdx + offsetY]) *
                             COORDINATE_PARAM - MEAN_PARAM) / layer.height;
            float width = (fastmath::sigmoid(static_cast<float *>(netout.get())[bIdx + OFFSETWIDTH]) *
                           COORDINATE_PARAM) * (fastmath::sigmoid(static_cast<float *>(netout.get())[bIdx +
                                                                                                     OFFSETWIDTH]) *
                                                COORDINATE_PARAM) * layer.anchors[BIASESDIM * k] / info.netWidth;
            float height = (fastmath::sigmoid(static_cast<float *>(netout.get())[bIdx + OFFSETHEIGHT]) *
                            COORDINATE_PARAM) * (fastmath::sigmoid(static_cast<float *>(netout.get())[bIdx +
                                                                                                      OFFSETHEIGHT]) *
                                                 COORDINATE_PARAM) * layer.anchors[BIASESDIM * k + OFFSETBIASES] /
                           info.netHeight;
            CodeDuplicationSetDet(x, y, width, height);
            det.classId = classID;
            det.className = configData_.GetClassName(classID);
            det.confidence = maxProb;
            if (det.confidence < separateScoreThresh_[classID]) continue;
            detBoxes.emplace_back(det);
        }
    }
}

/*
 * @description: Select the highest confidence class label for each predicted box and save into detBoxes
 * @param netout  The feature data which contains box coordinates, objectness value and confidence of each class
 * @param info  Yolo layer info which contains class number, box dim and so on
 * @param detBoxes  ObjectInfo vector where all ObjectInfoes's confidences are greater than threshold
 * @param stride  Stride of output feature data
 * @param layer  Yolo output layer
 */
/*
 * @description:为每个预测框选择置信度最高的类名
 * @param netout 特征数据，包括盒子坐标、目标值和每个类的置信度
 * @param info Yolo Yolo层信息，包含类号、框尺寸等
 * @param detBoxes ObjectInfo容器，其中所有ObjectInfo的置信度都大于阈值
 * @param stride 输出特征数据的步长
 * @param layer Yolo输出层
 */
void RefineDetPostProcess::SelectClassNHWC(std::shared_ptr<void> netout, NetInfo info,
                                           std::vector<MxBase::ObjectInfo> &detBoxes, int stride, OutputLayer layer) {
    const int offsetY = 1;
    for (int j = 0; j < stride; ++j) {
        for (int k = 0; k < info.anchorDim; ++k) {
            // begin index
            int bIdx = (info.bboxDim + 1 + info.classNum) * info.anchorDim * j +
                       k * (info.bboxDim + 1 + info.classNum);
            // objectness index
            int oIdx = bIdx + info.bboxDim;
            // check obj
            float objectness = fastmath::sigmoid(static_cast<float *>(netout.get())[oIdx]);
            if (objectness <= objectnessThresh_) {
                continue;
            }
            int classID = -1;
            float maxProb = scoreThresh_;
            float classProb;
            // Compare the confidence of the 3 anchors, select the largest one
            for (int c = 0; c < info.classNum; ++c) {
                classProb = fastmath::sigmoid(static_cast<float *>
                        (netout.get())[bIdx + (info.bboxDim + OFFSETOBJECTNESS + c)]) * objectness;
                CompareProb(classID, maxProb, classProb, c);
            }

            if (classID < 0) {continue;}
            MxBase::ObjectInfo det;
            int row = j / layer.width;
            int col = j % layer.width;
            // 计算检测对象的中心点
            float x = (col + fastmath::sigmoid(static_cast<float *>(netout.get())[bIdx])) / layer.width;
            float y = (row + fastmath::sigmoid(static_cast<float *>(netout.get())[bIdx + offsetY])) / layer.height;
            float width = fastmath::exp(static_cast<float *>(netout.get())[bIdx + OFFSETWIDTH]) *
                          layer.anchors[BIASESDIM * k] / info.netWidth;
            float height = fastmath::exp(static_cast<float *>(netout.get())[bIdx + OFFSETHEIGHT]) *
                           layer.anchors[BIASESDIM * k + OFFSETBIASES] / info.netHeight;
            // 根据中心计算对象框坐标
            det.x0 = std::max(0.0f, x - width / COORDINATE_PARAM);
            det.x1 = std::min(1.0f, x + width / COORDINATE_PARAM);
            det.y0 = std::max(0.0f, y - height / COORDINATE_PARAM);
            det.y1 = std::min(1.0f, y + height / COORDINATE_PARAM);

            det.classId = classID;
            det.className = configData_.GetClassName(classID);
            det.confidence = maxProb;
            if (det.confidence < separateScoreThresh_[classID]) {
                continue;
            }
            detBoxes.emplace_back(det);
        }
    }
}

/*
 * @description: According to the yolo layer structure, encapsulate the anchor box data of each feature into detBoxes
 * @param featLayerData  Vector of 3 output feature data
 * @param info  Yolo layer info which contains anchors dim, bbox dim, class number, net width, net height and
                3 outputlayer(eg. 13*13, 26*26, 52*52)
 * @param detBoxes  ObjectInfo vector where all ObjectInfoes's confidences are greater than threshold
 */
/*
 * @description:根据yolo层结构，将每个特征的锚盒数据封装到detBoxes中
 * @param featLayerData 包含3个输出特征数据的容器
 * @param detBoxes ObjectInfo容器，其中所有ObjectInfo的置信度都大于阈值
 */
void RefineDetPostProcess::GenerateBbox(std::vector<std::shared_ptr<void>> featLayerData,
                                        std::vector<MxBase::ObjectInfo> &detBoxes,
                                        const std::vector<std::vector<size_t>> &featLayerShapes, const int netWidth,
                                        const int netHeight) {
    NetInfo netInfo;
    netInfo.anchorDim = anchorDim_;
    netInfo.bboxDim = BOX_DIM;
    netInfo.classNum = classNum_;
    netInfo.netWidth = netWidth;
    netInfo.netHeight = netHeight;
    for (int i = 0; i < yoloType_; ++i) {
        int widthIndex_ = modelType_ ? NCHW_WIDTHINDEX : NHWC_WIDTHINDEX;
        int heightIndex_ = modelType_ ? NCHW_HEIGHTINDEX : NHWC_HEIGHTINDEX;
        // OutputLayer是RefineDet每个输出池对应的信息
        OutputLayer layer = {featLayerShapes[i][widthIndex_], featLayerShapes[i][heightIndex_]};

        int logOrder = log(featLayerShapes[i][widthIndex_] * SCALE / netWidth) / log(BIASESDIM);
        int startIdx = (yoloType_ - 1 - logOrder) * netInfo.anchorDim * BIASESDIM;
        int endIdx = startIdx + netInfo.anchorDim * BIASESDIM;
        int idx = 0;
        for (int j = startIdx; j < endIdx; ++j) {
            layer.anchors[idx++] = biases_[j];
        }

        int stride = layer.width * layer.height; // 13*13 26*26 52*52
        std::shared_ptr<void> netout = featLayerData[i];
        if (modelType_ == 0) {
            SelectClassNHWC(netout, netInfo, detBoxes, stride, layer);
        } else {
            if (yoloVersion_ == RefineDet_VERSION) {
                SelectClassNCHW(netout, netInfo, detBoxes, stride, layer);
            } else {
                SelectClassNCHWC(netout, netInfo, detBoxes, stride, layer);
            }
        }
    }
}

// 将biases字符串解析为int型数组存入biases_中
APP_ERROR RefineDetPostProcess::GetBiases(std::string &strBiases) {
    if (biasesNum_ <= 0) {
        LogError << GetError(APP_ERR_COMM_INVALID_PARAM) << "Failed to get biasesNum (" << biasesNum_ << ").";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    biases_.clear();
    int i = 0;
    int num = strBiases.find(",");
    while (num >= 0 && i < biasesNum_) {
        std::string tmp = strBiases.substr(0, num);
        num++;
        strBiases = strBiases.substr(num, strBiases.size());
        biases_.push_back(stof(tmp));
        i++;
        num = strBiases.find(",");
    }
    if (i != biasesNum_ - 1 || strBiases.size() <= 0) {
        LogError << GetError(APP_ERR_COMM_INVALID_PARAM) << "biasesNum (" << biasesNum_
                 << ") is not equal to total number of biases (" << strBiases << ").";
        return APP_ERR_COMM_INVALID_PARAM;
    }
    biases_.push_back(stof(strBiases));
    return APP_ERR_OK;
}
