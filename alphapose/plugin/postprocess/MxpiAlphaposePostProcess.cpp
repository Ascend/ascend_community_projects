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
#include "MxpiAlphaposePostProcess.h"
#include <numeric>
#include <algorithm>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "MxBase/Log/Log.h"
#include "MxBase/Tensor/TensorBase/TensorBase.h"

using namespace MxBase;
using namespace MxPlugins;
using namespace MxTools;
using namespace std;
using namespace cv;

namespace {
    auto g_uint8Deleter = [] (uint8_t *p) { };
    const float delta1 = 1;
    const float mu = 1.7;
    const float delta2 = 2.65;
    const float gma = 22.48;
    const float scoreThreds = 0.3;
    const float matchThreds = 5;
    const float areaThres = 0;
    const float alpha = 0.1;
    const float aspectRatio = 0.75;
    const int keypointsNum = 17;
    const int confidenceIndex = 0;
    const int centerxIndex = 1;
    const int centeryIndex = 2;
    const int scalewIndex = 3;
    const int scalehIndex = 4;
    bool accTest = false;
}

/**
 * @brief decode MxpiTensorPackageList
 * @param srcMxpiTensorPackageList - Source tensorPackageList
 * @param tensors - Target TensorBase data
 */
static void GetTensors(const MxTools::MxpiTensorPackageList &srcMxpiTensorPackageList,
                        std::vector<MxBase::TensorBase> &tensors)
{
    for (int i = 0; i < srcMxpiTensorPackageList.tensorpackagevec_size(); ++i) {
        MxTools::MxpiTensorPackage srcMxpiTensorPackage = srcMxpiTensorPackageList.tensorpackagevec(i);
        for (int j = 0; j < srcMxpiTensorPackage.tensorvec_size(); j++) {
            MxBase::MemoryData memoryData = {};
            memoryData.deviceId = srcMxpiTensorPackage.tensorvec(j).deviceid();
            memoryData.type = (MxBase::MemoryData::MemoryType)srcMxpiTensorPackage.tensorvec(j).memtype();
            memoryData.size = (uint32_t)srcMxpiTensorPackage.tensorvec(j).tensordatasize();
            memoryData.ptrData = (void *)srcMxpiTensorPackage.tensorvec(j).tensordataptr();
            std::vector<uint32_t> outputShape = {};
            for (int k = 0; k < srcMxpiTensorPackage.tensorvec(j).tensorshape_size(); ++k) {
                outputShape.push_back((uint32_t)srcMxpiTensorPackage.tensorvec(j).tensorshape(k));
            }
            MxBase::TensorBase tmpTensor(memoryData, true, outputShape,
                                         (MxBase::TensorDataType)srcMxpiTensorPackage.tensorvec(j).tensordatatype());
            tensors.push_back(tmpTensor);
        }
    }
}

/**
 * @brief decode MxpiObjectList
 * @param srcMxpiObjectList - Source MxpiObjectList
 * @param objectBoxes - The boxes of objects
 */
static void GetBoxes(const MxTools::MxpiObjectList &srcMxpiObjectList,
                        std::vector<std::vector<float> > &objectBoxes)
{
    int boxInfoNum = 5;
    for (int i = 0; i < srcMxpiObjectList.objectvec_size(); i++) {      
        MxTools::MxpiObject srcMxpiObject = srcMxpiObjectList.objectvec(i);
        // Filter out person class
        if ((accTest) || (srcMxpiObject.classvec(0).classid() == 0)){
            std::vector<float> objectBox(boxInfoNum);
            float x0 = srcMxpiObject.x0();            
            float y0 = srcMxpiObject.y0();            
            float x1 = srcMxpiObject.x1();            
            float y1 = srcMxpiObject.y1();           
            float centerx = (x1 + x0) * 0.5;
            float centery = (y1 + y0) * 0.5;       
            float boxWidth = x1 - x0;
            float boxHeight = y1 - y0;
            float confidence = 0;

            if (accTest){
                confidence = 1.0;
            }else{
                confidence = srcMxpiObject.classvec(0).confidence();
            }
            
            // Adjust the aspect ratio
            if (boxWidth >= aspectRatio * boxHeight){
                boxHeight = boxWidth / aspectRatio;
            }else{
                boxWidth = boxHeight * aspectRatio;
            }
            float scalew = boxWidth * 1.25;
            float scaleh = boxHeight * 1.25;

            objectBox[confidenceIndex] = confidence;
            objectBox[centerxIndex] = centerx;
            objectBox[centeryIndex] = centery;
            objectBox[scalewIndex] = scalew;
            objectBox[scalehIndex] = scaleh;
            objectBoxes.push_back(objectBox);
        }
    }
}

/**
 * @brief Parsing TensorBase data to keypoint heatmap and PAF heatmap of openpose model
 * @param tensors - TensorBase vector
 * @param result - std::vector<std::vector<cv::Mat> >
 */
static void ReadResultFromTensor(const std::vector <MxBase::TensorBase> &tensors, std::vector<std::vector<cv::Mat> > &result)
{
    const int heightIndex = 2, widthIndex = 3;
    //shape:1*17*64*48
    auto shape = tensors[0].GetShape();
    int keypointChannel = shape[1];
    int height = shape[heightIndex];
    int width = shape[widthIndex];
    for (int m = 0; m < tensors.size(); m++){
        // Read keypoint data
        auto dataPtr = (uint8_t *)tensors[m].GetBuffer();
        std::shared_ptr<void> keypointPointer;
        keypointPointer.reset(dataPtr, g_uint8Deleter);
        std::vector<cv::Mat> keypointHeatmap = {};
        int idx = 0;
        for (int i = 0; i < keypointChannel; i++) {
            cv::Mat singleChannelMat(height, width, CV_32FC1, cv::Scalar(0));
            for (int j = 0; j < height; j++) {
                float *ptr = singleChannelMat.ptr<float>(j);
                for (int k = 0; k < width;  k++) {
                    ptr[k] = static_cast<float *>(keypointPointer.get())[idx];
                    idx += 1;
                }
            }
            keypointHeatmap.push_back(singleChannelMat);
        }
        result.push_back(keypointHeatmap);
    }
}

/**
 * @brief Get the max predictive value
 * @param result - The model infer output
 * @param coords - Index of maximum values
 * @param maxvals - Maximum values
 */
static void GetMaxPrediction(const std::vector<std::vector<cv::Mat> > &result,
                            std::vector<cv::Mat> &coords, std::vector<cv::Mat> &maxvals)
{
    int channel = result[0].size();
    for (int i = 0; i < result.size(); i++){
        cv::Mat coord(channel, 2, CV_32FC1, cv::Scalar(0));
        cv::Mat maxval(channel, 1, CV_32FC1, cv::Scalar(0));
        for (int j = 0; j < channel; j++){
            // Get the max point's position and it's value
            double maxValue;
            cv::Point maxIdx;
            cv::minMaxLoc(result[i][j], NULL, &maxValue, NULL, &maxIdx); 
            float *ptr1 = coord.ptr<float>(j);
            float *ptr2 = maxval.ptr<float>(j);
            if (maxValue > 0){               
                ptr1[0] = (float)maxIdx.x;
                ptr1[1] = (float)maxIdx.y;               
            }else{
                ptr1[0] = (float)0;
                ptr1[1] = (float)0;               
            }          
            ptr2[0] = (float)maxValue;
        }
        coords.push_back(coord);
        maxvals.push_back(maxval);
    }
}

/**
 * @brief Get the third mapPoint for affine transform
 * @param mapPoint - cv::Point2f *
 */
static void GetThirdPoint(cv::Point2f *mapPoint)
{
    int thirdpointIndex = 2;
    float directx = mapPoint[0].x - mapPoint[1].x;
    float directy = mapPoint[0].y - mapPoint[1].y;
    mapPoint[thirdpointIndex].x = mapPoint[1].x - directy;
    mapPoint[thirdpointIndex].y = mapPoint[1].y + directx;
}

/**
 * @brief Get the transformation matrix for affine tranform
 * @param center - The center of object box
 * @param scale - The scale of objetc box
 * @param outputSize - The transformation matrix for affine tranform
 * @param trans - The transformation matrix for affine tranform
 */
static void GetAffineTransform(const std::vector<float> &center, const std::vector<float> &scale,
                                const std::vector<int> &outputSize, cv::Mat &trans)
{
    int pointNum = 3;
    cv::Point2f src[pointNum];
    src[0].x = center[0];
    src[0].y = center[1];
    src[1].x = center[0];
    src[1].y = center[1] - scale[0] * 0.5;
    GetThirdPoint(src);
    cv::Point2f dst[pointNum];
    dst[0].x = outputSize[0] * 0.5;
    dst[0].y = outputSize[1] * 0.5;
    dst[1].x = outputSize[0] * 0.5;
    dst[1].y = (outputSize[1] - outputSize[0]) * 0.5;
    GetThirdPoint(dst);
    // Get the transformation matrix for affine tranform
    trans = cv::getAffineTransform(dst, src);

}

/**
 * @brief Radiological transformation is performed to obtain the locations of keypoints on the original map
 * @param coords - Index of maximum values
 * @param objectBoxes - The boxes of objects
 * @param keypointPreds - Source data containing the information of keypoints position
 * @param heatmapWidth - The heatmap's width
 * @param heatmapHeight - The heatmap's height
 */
static void TransformPreds(const std::vector<cv::Mat> &coords, const std::vector<std::vector<float> > &objectBoxes,
                            std::vector<cv::Mat> &keypointPreds, int heatmapWidth, int heatmapHeight)
{
    
    std::vector<int> outputSize = {heatmapWidth, heatmapHeight};
    for (int i = 0; i < coords.size(); i++){
        std::vector<float> center = {};
        center.push_back(objectBoxes[i][centerxIndex]);
        center.push_back(objectBoxes[i][centeryIndex]);
        std::vector<float> scale = {};
        scale.push_back(objectBoxes[i][scalewIndex]);
        scale.push_back(objectBoxes[i][scalehIndex]);
        cv::Mat trans(2, 3, CV_32FC1, Scalar(0));
        
        // Get transformation matrix for affine transformation
        GetAffineTransform(center, scale, outputSize, trans);   

        // Radiological transformation
        cv::Mat coordT = coords[i].t();
        cv::Mat coordTExpansion(coords[i].cols + 1, coords[i].rows, CV_32FC1, Scalar(0));
        cv::copyMakeBorder(coordT, coordTExpansion, 0, 1, 0, 0, cv::BORDER_CONSTANT, 1.0);
        coordTExpansion.convertTo(coordTExpansion, CV_32FC1);
        trans.convertTo(trans, CV_32FC1);
        cv::Mat targetCoordT(coords[i].cols, coords[i].rows, CV_32FC1, Scalar(0));
        targetCoordT = trans * coordTExpansion;
        cv::Mat targetCoord = targetCoordT.t();
       
        keypointPreds.push_back(targetCoord);
    }   

}

/**
 * @brief Get parametric distance
 * @param pickId - Picked index
 * @param keypointPreds - Source data containing the information of keypoints position
 * @param keypointScores - Source data containing the information of keypoins score
 * @param finalDists - Final distance
 */
static void GetParametricDistance(int pickId, const std::vector<cv::Mat> &keypointPreds,
                                const std::vector<cv::Mat> &keypointScores, std::vector<float> &finalDists)
{
    int batchNum = keypointPreds.size();
    cv::Mat pickPred = keypointPreds[pickId].clone();
    cv::Mat predScore = keypointScores[pickId].clone();
    // Define a keypoints distance
    cv::Mat predScores(batchNum, predScore.rows, CV_32FC1);
    cv::Mat predScoreT = predScore.t();
    cv::copyMakeBorder(predScoreT, predScores, 0, (keypointsNum - 1), 0, 0, cv::BORDER_REPLICATE);

    for (int i = 0; i < batchNum; i++){
        float finalDist = 0;      
        for (int j =0; j < keypointPreds[i].rows; j++){
            const float *ptr1 = keypointPreds[i].ptr<float>(j);
            float *ptr2 = pickPred.ptr<float>(j);
            float p0 = powf((ptr2[0] - ptr1[0]), 2.0);
            float p1 = powf((ptr2[1] - ptr1[1]), 2.0);
            float p =p0 + p1;
            float dist = sqrt(p);
            float pointDist = exp((-1) * dist / delta2);          
            float scoreDist = 0.0;
            finalDist += (scoreDist + mu * pointDist);
        }    
        finalDists[i] = finalDist;
    }
    
}

/**
 * @brief Get numbers of match keypoints by calling PCKMatch
 * @param pickId - Picked index
 * @param refDist - Reference distance
 * @param keypointPreds - Source data containing the information of keypoints position
 * @param numMatchKeypoints - The numbers of match keypoints
 */
static void PCKMatch(int pickId, float refDist, const std::vector<cv::Mat> &keypointPreds,
                    std::vector<int> &numMatchKeypoints)
{
    int batchNum = keypointPreds.size();
    cv::Mat pickPred = keypointPreds[pickId].clone();

    for (int i = 0; i < batchNum; i++){
        int numMatchKeypoint = 0;
        for (int j =0; j < keypointPreds[i].rows; j++){
            const float *ptr1 = keypointPreds[i].ptr<float>(j);
            float *ptr2 = pickPred.ptr<float>(j);
            float p0 = powf((ptr2[0] - ptr1[0]), 2.0);
            float p1 = powf((ptr2[1] - ptr1[1]), 2.0);
            float p =p0 + p1;
            float dist = sqrt(p);
            refDist = std::min(refDist, (float)7.0);
            if ((dist / refDist) <= 1){
                numMatchKeypoint += 1;
            }else{
                numMatchKeypoint += 0;
            }
        }    
        numMatchKeypoints[i] = numMatchKeypoint;
    }

}

/**
 * @brief Merge poses
 * @param predsPick - The picked keypoints prediction
 * @param originkeypointPreds - The origin keypoints prediction
 * @param originkeypointScores - The origin keypoins score
 * @param refDist - Reference distance
 * @param mergeIds - The index of predicted keypoints to merge
 * @param mergePose - The merged pose
 * @param mergeScore - The merged score
 */
static void PoseMergeFast(const cv::Mat &predsPick, const std::vector<cv::Mat> &originKeypointPreds,
                        const std::vector<cv::Mat> &originKeypointScores, float refDist,
                        const std::vector<int> &mergeIds, cv::Mat &mergePose, cv::Mat &mergeScore)
{
    int kpNum = 17;
    int mergeSize = mergeIds.size();
    std::vector<cv::Mat> mergePs = {};
    std::vector<cv::Mat> mergeSs = {};
    // Get the keypoints required to merge from orgin keypoints
    for (int i = 0; i < mergeSize; i++){
        cv::Mat mergeP = originKeypointPreds[mergeIds[i]].clone();
        cv::Mat mergeS = originKeypointScores[mergeIds[i]].clone();
        mergePs.push_back(mergeP);
        mergeSs.push_back(mergeS);
    }

    std::vector<cv::Mat> maskScores = {};
    for (int i = 0; i < mergeSize; i++){ 
        cv::Mat maskScore(keypointsNum, 1, CV_32FC1, Scalar(0));   
        for (int j =0; j < keypointsNum; j++){
            float *ptr1 = mergePs[i].ptr<float>(j);
            const float *ptr2 = predsPick.ptr<float>(j);
            float *ptr3 = mergeSs[i].ptr<float>(j);
            float p0 = powf((ptr2[0] - ptr1[0]), 2.0);
            float p1 = powf((ptr2[1] - ptr1[1]), 2.0);
            float p =p0 + p1;
            float dist = std::sqrt(p);
            refDist = std::min(refDist, (float)15.0);
            float mask;
            if ((dist <= refDist)){
                mask = 1.0;
            }else{
                mask = 0.0;
            }
            float *ptr4 = maskScore.ptr<float>(j);
            ptr4[0] = ptr3[0] * mask;                               
        }   
        maskScores.push_back(maskScore); 
    }

    std::vector<float> sumScore(keypointsNum);
    for (int i = 0; i < keypointsNum; i++){
        float sum = 0;
        for (int j = 0; j < mergeSize; j++){
            float *ptr5 = maskScores[j].ptr<float>(i);
            sum += ptr5[0];
        }
        sumScore[i] = sum;
    }
    
    // Merge pose
    for (int i = 0; i < mergeSize; i++){
        cv::Mat tmpScore(keypointsNum, 1, CV_32FC1, Scalar(0));
        cv::Mat tmpPose(keypointsNum, 2, CV_32FC1, Scalar(0));
        for (int j = 0; j < keypointsNum; j++){
            float *ptr6 = maskScores[i].ptr<float>(j);
            float *ptr7 = mergePs[i].ptr<float>(j);
            float *ptr8 = mergeSs[i].ptr<float>(j);
            float normedScore = ptr6[0] / sumScore[j];
            float *ptr9 = tmpPose.ptr<float>(j);
            float *ptr10 = tmpScore.ptr<float>(j);
            ptr9[0] = ptr7[0] * normedScore;
            ptr9[1] = ptr7[1] * normedScore;
            ptr10[0] = ptr8[0] * normedScore;
        }
        cv::add(mergePose, tmpPose, mergePose);
        cv::add(mergeScore, tmpScore, mergeScore);
    }
}

/**
 * @brief Get final pose
 * @param maxValue - The max value of picked object's keypoints scores
 * @param confidencePick - The confidence of picked object
 * @param mergePose - The merged pose
 * @param mergeScore - The merged score
 * @param finalPoses - Target data containing the information of final keypoints' position
 * @param finalScores - Target data containing the information of fianl keypoints' score
 * @param personScores - Target data containing the information of person's score
 */
static void GetFinalPose(double maxValue, float confidencePick, cv::Mat &mergePose,
                        cv::Mat &mergeScore, std::vector<cv::Mat> &finalPoses,
                        std::vector<cv::Mat> &finalScores, std::vector<float> &personScores)
{       
    double maxValue1;
    cv::Point maxIdx1;
    cv::minMaxLoc(mergeScore, NULL, &maxValue1, NULL, &maxIdx1);
    if (maxValue1 > scoreThreds){
        cv::Mat mergePoseX = mergePose.colRange(0, 1).clone();
        cv::Mat mergePoseY = mergePose.colRange(1, 2).clone();
        double minValue2, maxValue2;
        cv::Point minIdx2, maxIdx2;
        cv::minMaxLoc(mergePoseX, &minValue2, &maxValue2, &minIdx2, &maxIdx2);
        double xLeft = minValue2;
        double xRight = maxValue2;            
        double minValue3, maxValue3;
        cv::Point minIdx3, maxIdx3;
        cv::minMaxLoc(mergePoseX, &minValue3, &maxValue3, &minIdx3, &maxIdx3);
        double yLeft = minValue3;
        double yRight = maxValue3;              
        if (2.15 * (xRight - xLeft) * (yRight - yLeft) > areaThres){
            finalPoses.push_back(mergePose.clone());
            finalScores.push_back(mergeScore.clone());
            float personScore = cv::mean(mergeScore)[0] + confidencePick + 1.25 * (float)maxValue;
            personScores.push_back(personScore);
        }
    }   
}

/**
    * @brief Extract keypoints' location information and scores
    * @param result - Source data containing the information of heatmap data
    * @param objectBoxes - Source data containing the information of object
    * @param keypointPreds - Source data containing the information of keypoints position
    * @param keypointScores - Source data containing the information of keypoins score
    * @return APP_ERROR
 */
APP_ERROR MxpiAlphaposePostProcess::ExtractKeypoints(const std::vector<std::vector<cv::Mat> > &result,
                                                    const std::vector<std::vector<float> > &objectBoxes,
                                                    std::vector<cv::Mat> &keypointPreds,
                                                    std::vector<cv::Mat> &keypointScores)
{
    // Get the max predictive value
    // coords: n*17*2   maxvals: n*17*1
    std::vector<cv::Mat> coords;
    std::vector<cv::Mat> maxvals;
    GetMaxPrediction(result, coords, maxvals);

    int heatMapHeight = result[0][0].rows;
    int heatMapWidth = result[0][0].cols;
    for (int i = 0; i < coords.size(); i++){
        for (int j = 0; j < coords[i].rows; j++){
            float *ptr = coords[i].ptr<float>(j);
            cv::Mat heatMap = result[i][j];
            
            int px = (int)floor(ptr[0] + 0.5);
            int py = (int)floor(ptr[1] + 0.5);
            if ((px > 1)&&(px < heatMapWidth-1)&&(py > 1)&&(py < heatMapHeight-1)){
                float diff1 = heatMap.at<float>(py,px+1) - heatMap.at<float>(py,px-1);           
                if (diff1 > 0){
                    ptr[0] += 0.25; 
                }else if (diff1 < 0){
                    ptr[0] -= 0.25;
                }
                float diff2 = heatMap.at<float>(py+1,px) - heatMap.at<float>(py-1,px);               
                if (diff2 > 0){
                    ptr[1] += 0.25; 
                }else if (diff2 < 0){
                    ptr[1] -= 0.25;
                }
            }
        }
    }

    // Transform the keypoint position from heatmap to origin image
    TransformPreds(coords, objectBoxes, keypointPreds, heatMapWidth, heatMapHeight);
    
    keypointScores.assign(maxvals.begin(), maxvals.end());
    return APP_ERR_OK;
}

/**
 * @brief Do maximum suppression to remove redundant keypoints' information
 * @param keypointPreds - Source data containing the information of keypoints position
 * @param keypointScores - Source data containing the information of keypoins score
 * @param objectBoxes - Source data containing the information of object
 * @param finalPoses - Target data containing the information of final keypoints' position
 * @param finalScores - Target data containing the information of fianl keypoints' score
 * @param personScores - Target data containing the information of person's score
 * @return APP_ERROR
*/
APP_ERROR MxpiAlphaposePostProcess::PoseNms(std::vector<cv::Mat> &keypointPreds,
                                            std::vector<cv::Mat> &keypointScores,
                                            std::vector<std::vector<float> > &objectBoxes,
                                            std::vector<cv::Mat> &finalPoses,
                                            std::vector<cv::Mat> &finalScores,
                                            std::vector<float> &personScores)
{
    int batchNum = keypointScores.size();
    std::vector<float> confidence(batchNum);
    std::vector<float> boxWidth(batchNum);
    std::vector<float> boxHeight(batchNum);
    std::vector<float> refDists(batchNum);
    std::vector<float> humanScores(batchNum);
    std::vector<int> humanIds(batchNum);
    std::vector<float> finalDists(batchNum);
    std::vector<int> numMatchKeypoints(batchNum);  
    for (int i = 0; i < batchNum; i++){
        for (int j = 0; j < keypointScores[i].rows; j++){
            float *ptr = keypointScores[i].ptr<float>(j);
            if (ptr[0] == 0.0){
                ptr[0] == 1e-5;
            }
        }
        confidence[i] = objectBoxes[i][confidenceIndex];
        boxWidth[i] = objectBoxes[i][scalewIndex] * 0.8;
        boxHeight[i] = objectBoxes[i][scalehIndex] * 0.8;
        if (boxHeight[i] >= boxWidth[i]){
            refDists[i] = alpha * boxHeight[i];
        }else  {
            refDists[i] = alpha * boxWidth[i];
        }
        humanScores[i] = cv::mean(keypointScores[i])[0];
        humanIds[i] = i;      
    }

    // Do pPose-NMS
    std::vector<cv::Mat> originKeypointPreds(keypointPreds);
    std::vector<cv::Mat> originKeypointScores(keypointScores);
    std::vector<float> originconfidence(confidence);
    while (humanScores.size() != 0){
        int tmpSize = humanScores.size();
        // Pick the one with highest score
        std::vector<float>::iterator biggest = std::max_element(std::begin(humanScores), std::end(humanScores));
        int pickId = std::distance(std::begin(humanScores), biggest);
        cv::Mat predsPick = originKeypointPreds[humanIds[pickId]];
        cv::Mat scoresPick = originKeypointScores[humanIds[pickId]];
        float confidencePick = originconfidence[humanIds[pickId]];  
        // Get numbers of match keypoints by calling PCK_match
        float refDist = refDists[humanIds[pickId]];
        GetParametricDistance(pickId, keypointPreds, keypointScores, finalDists);
        PCKMatch(pickId, refDist, keypointPreds, numMatchKeypoints);
        // Delete humans who have more than matchThreds keypoints overlap and high similarity
        std::vector<int> mergeIds = {};
        std::vector<int> deleteIds = {};
        int count = 0;
        for (int i = 0; i < tmpSize; i++){           
            if ((finalDists[i] > gma) || (numMatchKeypoints[i] > matchThreds)){  
                int deleteId = i - count;
                int mergeId = humanIds[deleteId];
                mergeIds.push_back(mergeId);
                keypointPreds.erase(keypointPreds.begin() + deleteId);
                keypointScores.erase(keypointScores.begin() + deleteId);
                humanIds.erase(humanIds.begin() + deleteId);
                humanScores.erase(humanScores.begin() + deleteId);
                confidence.erase(confidence.begin() + deleteId);     
                count ++;           
            }        
        }
        if (mergeIds.size() == 0){
            int deleteId = pickId;
            int mergeId = humanIds[deleteId];
            mergeIds.push_back(mergeId); 
            keypointPreds.erase(keypointPreds.begin() + deleteId);
            keypointScores.erase(keypointScores.begin() + deleteId);
            humanIds.erase(humanIds.begin() + deleteId);
            humanScores.erase(humanScores.begin() + deleteId);
            confidence.erase(confidence.begin() + deleteId);            
        }

        double maxValue;
        cv::Point maxIdx;
        cv::minMaxLoc(scoresPick, NULL, &maxValue, NULL, &maxIdx);
        if (maxValue >= scoreThreds){           
            cv::Mat mergePose(keypointsNum, 2, CV_32FC1, Scalar(0));
            cv::Mat mergeScore(keypointsNum, 1, CV_32FC1, Scalar(0));
            if (mergeIds.size() == 1){
                mergePose = originKeypointPreds[mergeIds[0]].clone();
                mergeScore = originKeypointScores[mergeIds[0]].clone();
            }else{   
                PoseMergeFast(predsPick, originKeypointPreds, originKeypointScores, 
                            refDist, mergeIds, mergePose, mergeScore);
            }       
            GetFinalPose(maxValue, confidencePick, mergePose, 
                        mergeScore, finalPoses, finalScores, personScores);        
        }
    }
    return APP_ERR_OK;
}

/**
 * @brief Prepare output in the format of MxpiPersonList
 * @param finalPoses - Source data containing the information of final keypoints' position
 * @param finalScores - Source data containing the information of fianl keypoints' score
 * @param dstMxpiPersonList - Target data in the format of MxpiPersonList
 * @return APP_ERROR
*/
APP_ERROR MxpiAlphaposePostProcess::GenerateMxpiOutput(std::vector<cv::Mat> &finalPoses,
                                                        std::vector<cv::Mat> &finalScores,
                                                        std::vector<float> &personScores,
                                                        mxpialphaposeproto::MxpiPersonList &dstMxpiPersonList)
{
    for (int i = 0; i < finalPoses.size(); i++) {
        auto mxpiPersonPtr = dstMxpiPersonList.add_personinfovec();
        mxpialphaposeproto::MxpiMetaHeader *dstPersonMxpiMetaheaderList = mxpiPersonPtr->add_headervec();
        dstPersonMxpiMetaheaderList->set_datasource(parentName_);
        dstPersonMxpiMetaheaderList->set_memberid(0);
        mxpiPersonPtr->set_confidence(personScores[i]);
        for (int j = 0; j < finalPoses[i].rows; j++){
            auto mxpiKeypointPtr = mxpiPersonPtr->add_keypoints();
            float *ptr1 = finalPoses[i].ptr<float>(j);
            float *ptr2 = finalScores[i].ptr<float>(j);
            mxpiKeypointPtr->set_x(ptr1[0]);
            mxpiKeypointPtr->set_y(ptr1[1]);
            mxpiKeypointPtr->set_score(ptr2[0]);
        }        
    }
    return APP_ERR_OK;
}

/**
 * @brief Overall process to generate all person keypoints information
 * @param srcMxpiObjectList - Source MxpiObjectList containing object data about input image
 * @param srcMxpiTensorPackageList - Source MxpiTensorPackageList containing heatmap data
 * @param dstMxpiPersonList - Target MxpiPersonList containing detection result list
 * @return APP_ERROR
*/
APP_ERROR MxpiAlphaposePostProcess::GeneratePersonList(const MxpiObjectList &srcMxpiObjectList,
                                                      const MxpiTensorPackageList &srcMxpiTensorPackageList,
                                                      mxpialphaposeproto::MxpiPersonList &dstMxpiPersonList)
{
    // Get object boxes from object detector
    // objectBoxes: n*5, n is the number of person
    std::vector<std::vector<float> > objectBoxes = {};
    GetBoxes(srcMxpiObjectList, objectBoxes);
    std::vector<cv::Mat> finalPoses = {};
    std::vector<cv::Mat> finalScores = {};
    std::vector<float> personScores = {};
    if (objectBoxes.size() == 0){
        cv::Mat finalPose(keypointsNum, 2, CV_32FC1, Scalar(0));
        cv::Mat finalScore(keypointsNum, 1, CV_32FC1, Scalar(0));
        float personScore = 0.0;
        finalPoses.push_back(finalPose);
        finalScores.push_back(finalScore);
        personScores.push_back(personScore);
    }else{
        // Get tensor from tensorinfer output
        std::vector<MxBase::TensorBase> tensors = {};
        GetTensors(srcMxpiTensorPackageList, tensors);
        
        // Read data from tensor output by the upstream plugin
        // result: n*17*64*48, n is the number of person
        std::vector<std::vector<cv::Mat> > result = {};
        ReadResultFromTensor(tensors, result);

        // Get The keypoints and their scores
        std::vector<cv::Mat> keypointPreds = {};
        std::vector<cv::Mat> keypointScores = {};
        ExtractKeypoints(result, objectBoxes, keypointPreds, keypointScores);       
        // DO pose nms
        if (accTest){         
            for (int i = 0; i < keypointPreds.size(); i++){
                double maxValue;
                cv::Point maxIdx;
                cv::minMaxLoc(keypointScores[i], NULL, &maxValue, NULL, &maxIdx);
                GetFinalPose(maxValue, objectBoxes[i][0], keypointPreds[i], 
                            keypointScores[i], finalPoses, finalScores, personScores);
            }
        }else{
            PoseNms(keypointPreds, keypointScores, objectBoxes, finalPoses, finalScores, personScores);
        }
    }
 
    // Prepare output in the format of MxpiPersonList
    GenerateMxpiOutput(finalPoses, finalScores, personScores, dstMxpiPersonList);
    return APP_ERR_OK;
}

/**
 * @brief Initialize configure parameter.
 * @param configParamMap
 * @return APP_ERROR
 */
APP_ERROR MxpiAlphaposePostProcess::Init(std::map<std::string, std::shared_ptr<void>> &configParamMap)
{
    LogInfo << "MxpiAlphaposePostProcess::Init start.";
    APP_ERROR ret = APP_ERR_OK;
    // Get the property values by key
    std::shared_ptr<string> parentNamePropSptr = std::static_pointer_cast<string>(configParamMap["dataSource"]);
    parentName_ = *parentNamePropSptr.get();
    std::shared_ptr<string> objectDetectorPropSptr = std::static_pointer_cast<string>(configParamMap["objectSource"]);
    objectDetectorName_ = *objectDetectorPropSptr.get();
    return APP_ERR_OK;
}

/**
 * @brief DeInitialize configure parameter.
 * @return APP_ERROR
 */
APP_ERROR MxpiAlphaposePostProcess::DeInit()
{
    LogInfo << "MxpiAlphaposePostProcess::DeInit end.";
    LogInfo << "MxpiAlphaposePostProcess::DeInit end.";
    return APP_ERR_OK;
}

/**
 * @brief Process the data of MxpiBuffer.
 * @param mxpiBuffer
 * @return APP_ERROR
 */
APP_ERROR MxpiAlphaposePostProcess::Process(std::vector<MxpiBuffer*> &mxpiBuffer)
{
    MxpiBuffer *buffer = mxpiBuffer[0];
    MxpiMetadataManager mxpiMetadataManager(*buffer);
    MxpiErrorInfo mxpiErrorInfo;
    ErrorInfo_.str("");
    auto errorInfoPtr = mxpiMetadataManager.GetErrorInfo();
    if (errorInfoPtr != nullptr) {
        ErrorInfo_ << GetError(APP_ERR_COMM_FAILURE, pluginName_) <<
        "MxpiAlphaposePostProcess process is not implemented";
        mxpiErrorInfo.ret = APP_ERR_COMM_FAILURE;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        LogError << "MxpiAlphaposePostProcess process is not implemented";
        return APP_ERR_COMM_FAILURE;
    }
    // Get the output of tensorinfer from buffer
    shared_ptr<void> tensorMetadata = mxpiMetadataManager.GetMetadata(parentName_);
    if (tensorMetadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "tensor metadata is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_METADATA_IS_NULL; 
    }
    shared_ptr<MxpiTensorPackageList> srcMxpiTensorPackageListSptr
	    = static_pointer_cast<MxpiTensorPackageList>(tensorMetadata);

    MxpiTensorPackageList srcMxpiTensorPackageList = *srcMxpiTensorPackageListSptr;
    // Get the output of objectdetector from buffer
    if (objectDetectorName_ == "appInput"){
        accTest = true;
        
    }
    shared_ptr<void> objectMetadata = mxpiMetadataManager.GetMetadata(objectDetectorName_);
    if (objectMetadata == nullptr) {
        ErrorInfo_ << GetError(APP_ERR_METADATA_IS_NULL, pluginName_) << "objectDetector metadata is NULL, failed";
        mxpiErrorInfo.ret = APP_ERR_METADATA_IS_NULL;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return APP_ERR_METADATA_IS_NULL; 
    }
    shared_ptr<MxpiObjectList> srcMxpiObjectListSptr
	    = static_pointer_cast<MxpiObjectList>(objectMetadata);
    MxpiObjectList srcMxpiObjectList = *srcMxpiObjectListSptr;

    // Generate output
    shared_ptr<mxpialphaposeproto::MxpiPersonList> dstMxpiPersonListSptr =
            make_shared<mxpialphaposeproto::MxpiPersonList>();
    APP_ERROR ret = GeneratePersonList(srcMxpiObjectList, srcMxpiTensorPackageList, *dstMxpiPersonListSptr);
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiAlphaposePostProcess get person's keypoint information failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    ret = mxpiMetadataManager.AddProtoMetadata(pluginName_, static_pointer_cast<void>(dstMxpiPersonListSptr));
    if (ret != APP_ERR_OK) {
        ErrorInfo_ << GetError(ret, pluginName_) << "MxpiAlphaposePostProcess add metadata failed.";
        mxpiErrorInfo.ret = ret;
        mxpiErrorInfo.errorInfo = ErrorInfo_.str();
        SetMxpiErrorInfo(*buffer, pluginName_, mxpiErrorInfo);
        return ret;
    }
    // Send the data to downstream plugin
    SendData(0, *buffer);
    return APP_ERR_OK;
}

/**
 * @brief Definition the parameter of configure properties.
 * @return std::vector<std::shared_ptr<void>>
 */
std::vector<std::shared_ptr<void>> MxpiAlphaposePostProcess::DefineProperties()
{
    std::vector<std::shared_ptr<void>> properties;
    // Set the type and related information of the properties, and the key is the name
    auto parentNameProSptr = (std::make_shared<ElementProperty<string>>)(ElementProperty<string>{
            STRING, "dataSource", "parentName", "the name of previous plugin", "mxpi_tensorinfer1", "NULL", "NULL"});
    auto objectDetectorProSptr = (std::make_shared<ElementProperty<string>>)(ElementProperty<string>{
            STRING, "objectSource", "objectName", "the name of object postprocess plugin", "mxpi_objectpostprocessor0", "NULL", "NULL"});
    properties.push_back(parentNameProSptr);
    properties.push_back(objectDetectorProSptr);

    return properties;
}

APP_ERROR MxpiAlphaposePostProcess::SetMxpiErrorInfo(MxpiBuffer &buffer, const std::string plugin_name,
                                                    const MxpiErrorInfo mxpiErrorInfo)
{
    APP_ERROR ret = APP_ERR_OK;
    // Define an object of MxpiMetadataManager
    MxpiMetadataManager mxpiMetadataManager(buffer);
    ret = mxpiMetadataManager.AddErrorInfo(plugin_name, mxpiErrorInfo);
    if (ret != APP_ERR_OK) {
        LogError << "Failed to AddErrorInfo.";
        return ret;
    }
    ret = SendData(0, buffer);
    return ret;
}

// Register the plugin through macro
MX_PLUGIN_GENERATE(MxpiAlphaposePostProcess)
