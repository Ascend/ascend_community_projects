#!/usr/bin/env python
# coding=utf-8

"""

 Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

"""
import json
import sys
import stat
import os

import cv2
import numpy as np

from StreamManagerApi import StreamManagerApi, StringVector, MxDataInput
from utils.visualization import visualize
sys.path.append("../proto/")
import mxpiAlphaposeProto_pb2 as mxpialphaposeproto



def main():
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # Creat stream
    path = b"../pipeline/image.pipeline"
    ret = stream_manager_api.CreateMultipleStreamsFromFile(path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    data_input = MxDataInput()
    stream_name = b'alphapose'
    in_plugin_id = 0
    image_dir = '../data'
    if os.path.exists(image_dir) != 1:
        print("Please create a folder for your test pictures.")
        exit()
    if len(os.listdir(image_dir)) ==0:
        print("There are no test pictures in the folder.")
        exit()
    for file_path in os.listdir(image_dir):
        image_name = file_path
        file_path = os.path.join(image_dir, file_path)
        # Construct the input of the stream
        with open(file_path, 'rb') as f:
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
        personlist = []
        for i, _ in enumerate(range(person_num)):
            person = pose_out_list.personInfoVec[i]
            keypoints_score = np.zeros((17, 1), dtype = np.float32)
            keypoints_pre = np.zeros((17, 2), dtype = np.float32)
            for j in range(len(person.keyPoints)):
                keypoints_score[j][0] = person.keyPoints[j].score
                keypoints_pre[j][0] = person.keyPoints[j].x
                keypoints_pre[j][1] = person.keyPoints[j].y
            score = np.array(person.confidence)
            personlist.append({
                'keypoints': keypoints_pre,
                'kp_score': keypoints_score,
                'proposal_score': score
            })
        # Read the original image
        origin_image = cv2.imread(file_path)
        # Visual keypoints
        vis_image = visualize(origin_image, personlist)
        # Save the picture
        cv2.imwrite("../out/{}".format(image_name), vis_image)
        # Save key point information to JSON file
        for i, _ in enumerate(range(person_num)):
            personlist[i]['keypoints'] = personlist[i]['keypoints'].tolist()
            personlist[i]['kp_score'] = personlist[i]['kp_score'].tolist()
            personlist[i]['proposal_score'] = personlist[i]['proposal_score'].tolist()
        flags = os.O_WRONLY | os.O_CREAT # Sets how files are read and written
        modes = stat.S_IWUSR | stat.S_IRUSR  # Set file permissions
        json_file = "../out/{}.json".format(image_name)
        with os.fdopen(os.open(json_file, flags, modes), 'w') as f:
            json.dump(personlist, f, indent=2)
        print('result was written successfully')

    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    main()