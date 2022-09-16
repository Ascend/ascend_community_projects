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
import stat
import os
import time
from StreamManagerApi import StreamManagerApi, MxDataInput


QUERY_STREAM_NAME = b'queryImageProcess'

IN_PLUGIN_ID = 0
OUT_PLUGIN_ID = 0


def initialize_stream():
    """
    Initialize stream queryImageProcess for detecting and re-identifying persons in image

    :arg:
        None

    :return:
        Stream api
    """
    
    stream_pi = StreamManagerApi()
    ret = stream_pi.InitManager()
    if ret != 0:
        error_message = "Failed to init Stream manager, ret=%s" % str(ret)
        print(error_message)
        exit()

    # creating stream based on json strings in the pipeline file: 'ReID.pipeline'
    with open("pipeline/image.pipeline", 'rb') as f:
        pipeline = f.read()

    ret = stream_pi.CreateMultipleStreams(pipeline)
    if ret != 0:
        error_message = "Failed to create Stream, ret=%s" % str(ret)
        print(error_message)
        exit()

    return stream_pi


def reid(input_dir, output_dir, stream_api):
    
    """
     detecting and re-identifying persons in videos
     
    :arg:
        inputDir: the directory of query images
        outputDir: the directory of output images
        streamApi: stream api
        
    :return:
        None
    """
        
    if os.path.exists(input_dir) != 1:
        error_message = 'The img dir does not exist.'
        print(error_message)
        exit()
    if len(os.listdir(input_dir)) == 0:
        error_message = 'The img file is empty.'
        print(error_message)
        exit()
    if os.path.exists(output_dir) != 1:
        root = os.getcwd()
        os.makedirs(os.path.join(root, output_dir))
        
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.JPG'):
                data_input = MxDataInput()
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    data_input.data = f.read()
                start = time.time()
                
                unique_id = stream_api.SendData(QUERY_STREAM_NAME, IN_PLUGIN_ID, data_input)
                if unique_id < 0:
                    error_message = 'Failed to send data to queryImageProcess stream.'
                    print(error_message)
                    exit()
                    
                # get infer result
                infer_result = stream_api.GetResult(QUERY_STREAM_NAME, OUT_PLUGIN_ID)
                if infer_result.errorCode:
                    error_message = 'Unable to get effective infer results, please check the stream log for details'
                    print(error_message)
                    exit()
                end = time.time()    
                out_path = os.path.join(output_dir, file)
                flags = os.O_WRONLY | os.O_CREAT # Sets how files are read and written
                modes = stat.S_IWUSR | stat.S_IRUSR  # Set file permissions
                with os.fdopen(os.open(out_path, flags, modes), 'wb') as f:
                    f.write(infer_result.data)
                print("time:", end - start)
            else:
                print('Input image only support jpg')
                exit()
   
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--queryInputDir', type=str, default='data/query/images', help="Query Images Input Dir")
    parser.add_argument('--queryOutputDir', type=str, default='output/query/images', help="Query Images Output Dir")
    opt = parser.parse_args()
    stream_manager_api = initialize_stream()
    reid(opt.queryInputDir, opt.queryOutputDir, stream_manager_api)
    
    stream_manager_api.DestroyAllStreams()