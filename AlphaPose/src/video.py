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
import argparse
import json
import os
import sys
import stat
import time

import cv2
import numpy as np

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, StringVector
from utils.visualization import visualize, visualize_fast
sys.path.append("../proto/")
import mxpiAlphaposeProto_pb2 as mxpialphaposeproto


DECODE_INDEX = 0
POSE_INDEX = 1
VIDEO_WIDTH = 720
VIDEO_HEIGHT = 1280
YUV_BYTES_NU = 3
YUV_BYTES_DE = 2

parser = argparse.ArgumentParser(description='AlphaPose')
parser.add_argument('--speedtest', default=False, action='store_true', help='The test frame rate')


def main():
    args = parser.parse_args()
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    # Creat stream
    path = b"../pipeline/video.pipeline"
    ret = stream_manager_api.CreateMultipleStreamsFromFile(path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    start = time.time()

    stream_name = b'alphapose'
    key_vec = StringVector()
    key_vec.push_back(b"mxpi_videodecoder0")
    key_vec.push_back(b"mxpi_alphaposepostprocess0")

    if (VIDEO_WIDTH == 0) or (VIDEO_HEIGHT == 0):
        print("The width and height of the input and output video should not be zero")
        exit()
    # Example Initialize the video encoder
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # H.264 codec
    out = cv2.VideoWriter(filename="../out/alphapose.avi", fourcc=fourcc, fps=24,
                        frameSize=(VIDEO_WIDTH, VIDEO_HEIGHT), isColor=True)
    frame_count = 0
    wait_count = 0
    while True:
        infer_result = stream_manager_api.GetResult(stream_name, b'appsink0', key_vec)
        if infer_result.errorCode != 0:
            if wait_count != 10:
                print("Please check the rtspUrl of the video is correct or the video exists.")
                wait_count += 1
                continue
            else:
                break

        # Obtain the results of videodecoder
        vision_list = MxpiDataType.MxpiVisionList()
        vision_list.ParseFromString(infer_result.metadataVec[DECODE_INDEX].serializedMetadata)
        vision_data = vision_list.visionVec[0].visionData.dataStr
        vision_info = vision_list.visionVec[0].visionInfo

        img_yuv = np.frombuffer(vision_data, dtype = np.uint8)
        img_yuv = img_yuv.reshape(vision_info.heightAligned * YUV_BYTES_NU // YUV_BYTES_DE, vision_info.widthAligned)
        img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR_NV12)

        # Obtain the post-processing results of key point detection
        pose_out_list = mxpialphaposeproto.MxpiPersonList()
        pose_out_list.ParseFromString(infer_result.metadataVec[POSE_INDEX].serializedMetadata)
        person_num = len(pose_out_list.personInfoVec)
        personlist = []
        for i, _ in enumerate(range(person_num)):
            person = pose_out_list.personInfoVec[i]
            keypoints_num = 17
            keypoints_score = np.zeros((keypoints_num, 1), dtype = np.float32)
            keypoints_pre = np.zeros((keypoints_num, 2), dtype = np.float32)
            for j in range(len(person.keyPoints)):
                keypoints_score[j][0] = person.keyPoints[j].score
                keypoints_pre[j][0] = person.keyPoints[j].x
                keypoints_pre[j][1] = person.keyPoints[j].y
            score = np.array(person.confidence)
            personlist.append({
                'frame': frame_count,
                'keypoints': keypoints_pre,
                'keypoints_score': keypoints_score,
                'proposal_score': score,
            })
        # Whether to conduct speed tests
        if not args.speedtest:
            # visualize and save
            img = visualize(img_bgr, personlist)
            vis_image = img[:VIDEO_HEIGHT, :VIDEO_WIDTH, :]
            out.write(vis_image)
        # Save key point information to JSON file
        for i, _ in enumerate(range(person_num)):
            personlist[i]['keypoints'] = personlist[i]['keypoints'].tolist()
            personlist[i]['keypoints_score'] = personlist[i]['keypoints_score'].tolist()
            personlist[i]['proposal_score'] = personlist[i]['proposal_score'].tolist()
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT  # Sets how files are read and written
        modes = stat.S_IWUSR | stat.S_IRUSR  # Set file permissions
        json_file = "../out/alphapose.json"
        with os.fdopen(os.open(json_file, flags, modes), 'a') as f:
            json.dump(personlist, f, indent=2)
        frame_count += 1
        # The time and frame rate information is printed every 10 frames
        if frame_count % 10 == 0:
            end = time.time()
            cost_time = end - start
            print("*******************************************************************")
            print("Frame count:%d" % frame_count)
            print("Spend time:%10.3f" % cost_time)
            print("fps:%10.3f" % (10 / cost_time ))
            print("*******************************************************************")
            start = time.time()

    out.release()
    stream_manager_api.DestroyAllStreams()


if __name__ == '__main__':
    main()