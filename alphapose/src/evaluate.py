#!/usr/bin/env python
# coding=utf-8

# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
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
import sys
import os
import json
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector, RoiBox, RoiBoxVector
sys.path.append("../proto")
import mxpiAlphaposeProto_pb2 as mxpialphaposeproto


def main():
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    path = b"../pipeline/evaluate.pipeline"
    ret = stream_manager_api.CreateMultipleStreamsFromFile(path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    image_folder = 'dataset/val2017'
    annotation_file = 'dataset/annotations/person_keypoints_val2017.json'
    detect_file = 'val2017_keypoint_detect_result.json'
    
    coco = COCO(annotation_file)
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    image_list = annotations['images']

    stream_name = b'alphapose'
    in_plugin_id = 0
    data_input = MxDataInput()
    coco_result = []
    
    for image_idx, image_info in enumerate(image_list):   
        image_path = os.path.join(image_folder, image_info['file_name'])
        image_id = image_info['id']       
        ann_ids = coco.getAnnIds(image_id)
        anns = coco.loadAnns(ann_ids)
        roi_vector = RoiBoxVector()
        for i in range(len(anns)):
            roi = RoiBox()
            roi.x0 = anns[i]['bbox'][0]
            roi.y0 = anns[i]['bbox'][1]
            roi.x1 = anns[i]['bbox'][0] + anns[i]['bbox'][2]
            roi.y1 = anns[i]['bbox'][1] + anns[i]['bbox'][3]
            roi_vector.push_back(roi)
        
        data_input.roiBoxs = roi_vector
        print('Detect image: ', image_idx, ': ', image_info['file_name'], ', image id: ', image_id)        
        if os.path.exists(image_path) != 1:
            print("The image does not exist.")
            exit()
        with open(image_path, 'rb') as f:
            data_input.data = f.read()
        
        # Inputs data to a specified stream based on stream_name.
        ret = stream_manager_api.SendData(stream_name, in_plugin_id, data_input)
        if ret < 0:
            print("Failed to send data to stream.")
            exit()
        key_vec = StringVector()
        key_vec.push_back(b"mxpi_alphaposepostprocess0")

        infer_result = stream_manager_api.GetResult(stream_name, b'appsink0', key_vec)
        if infer_result.errorCode != 0:
            print("GetResult error. errorCode=%d, errorMsg=%s" % (infer_result.errorCode, infer_result.errorMsg))
            continue
    
        # Obtain the post-processing results of key point detection
        pose_out_list = mxpialphaposeproto.MxpiPersonList()
        pose_out_list.ParseFromString(infer_result.metadataVec[0].serializedMetadata)
        person_num = len(pose_out_list.personInfoVec)         
        for i in range(person_num):
            person = pose_out_list.personInfoVec[i]
            keypoints_score = np.zeros((17, 1), dtype = np.float32)
            keypoints_pre = np.zeros((17, 2), dtype = np.float32)
            for j in range(len(person.keyPoints)):
                keypoints_score[j][0] = person.keyPoints[j].score
                keypoints_pre[j][0] = person.keyPoints[j].x
                keypoints_pre[j][1] = person.keyPoints[j].y
            score = np.array(person.confidence)
            keypoints = np.concatenate((keypoints_pre, keypoints_score), axis=1)
            keypoints = keypoints.reshape(-1).tolist()

            data = dict()
            data['image_id'] = image_id
            data['score'] = score.tolist()
            data['category_id'] = 1
            data['keypoints'] = keypoints
            coco_result.append(data)
        
    with open(detect_file, 'w') as f:
        json.dump(coco_result, f, indent=4)
    # run coco evaluation process using COCO official evaluation tool
    annotation_type = 'keypoints'
    dtcoco = coco.loadRes(detect_file)
    result = COCOeval(coco, dtcoco, annotation_type)
    result.evaluate()
    result.accumulate()
    result.summarize()
    # destroy streams
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    main()
