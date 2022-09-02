#!/usr/bin/env python
# coding=utf-8

"""
Copyright(C) Huawei Technologies Co.,Ltd. 2012-2021 All rights reserved.

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

import argparse
import time
from StreamManagerApi import StreamManagerApi


def initialize_stream():
    """
    Initialize stream for detecting and re-identifying persons in video

    :arg:
        None

    :return:
        Stream api
    """
    stream_api = StreamManagerApi()
    ret = stream_api.InitManager()
    if ret != 0:
        error_message = "Failed to init Stream manager, ret=%s" % str(ret)
        print(error_message)
        exit()

    # creating stream based on json strings in the pipeline file: 'ReID.pipeline'
    with open("pipeline/video.pipeline", 'rb') as f:
        pipeline = f.read()

    ret = stream_api.CreateMultipleStreams(pipeline)
    if ret != 0:
        error_message = "Failed to create Stream, ret=%s" % str(ret)
        print(error_message)
        exit()

    return stream_api


def wait(duration):
    
    """
    Wait for destroy streams

    :arg:
        duration: Duration of process video
    
    :return:
        None
    """
    
    start = time.time()
    
    while True:
        now = time.time()
        cost_time = now - start
        if cost_time >= duration:
            break        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default= 30, help="Duration Of Process Video")
    opt = parser.parse_args()
    stream_manager_api = initialize_stream()
    wait(opt.duration)
    
    stream_manager_api.DestroyAllStreams()
