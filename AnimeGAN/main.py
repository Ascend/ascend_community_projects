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

import os
import glob
import time
import math
import cv2
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

# set some parameters
STREAM_NAME = b'animegan'
DATA_PATH = "HR_photo"
PIPELINE = "animegan.pipeline"


def preprocess(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]

    # Due to the limit of memory,the model doesn't support resolutions larger.
    h = min(max(384, h), 1536)
    w = min(max(384, w), 1536)

    # resize to align size to n*128,round up
    h = math.ceil(h / 128) * 128
    w = math.ceil(w / 128) * 128
    img = cv2.resize(img, (w, h))

    return cv2.imencode(".jpg", img)[1].tobytes()


if __name__ == '__main__':
    # check pipeline and dataset
    if not os.path.exists(PIPELINE):
        print("The pipeline does not exist.")
        exit()
    elif not os.path.exists(DATA_PATH):
        print("The test images don't exist.")
        exit()

    paths = sorted(glob.glob(os.path.join(DATA_PATH, "*.jpeg")) +
                   glob.glob(os.path.join(DATA_PATH, "*.jpg")))
    if len(paths) == 0:
        print("The dataset is empty!")
        exit()

    # initialize the stream manager
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by the pipeline config
    with open(PIPELINE, 'rb') as f:
        pipeline = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    for img_path in paths:
        # send data to stream
        dataInput = MxDataInput()
        dataInput.data = preprocess(img_path)
        ret = streamManagerApi.SendData(STREAM_NAME, b'appsrc0', dataInput)
        if ret < 0:
            print("Failed to send data to stream")
            exit()

        # get inference result
        key_vec = StringVector()
        key_vec.push_back(b'appsink0')
        infer_result = streamManagerApi.GetProtobuf(STREAM_NAME, 0, key_vec)

        # the output's filename is decided by timestamp,so better not let them overlap
        time.sleep(1)

    streamManagerApi.DestroyAllStreams()
