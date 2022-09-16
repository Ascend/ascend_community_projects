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
import os
import time
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector


GALLERY_STREAM_NAME = b'galleryProcess'
IN_PLUGIN_ID = 0
OUT_PLUGIN_ID = 0


def initialize_stream():
    """
    Initialize stream galleryImageProcess for detecting and re-identifying persons in galley images

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
    with open("pipeline/gallery.pipeline", 'rb') as f:
        pipeline = f.read()

    ret = stream_pi.CreateMultipleStreams(pipeline)
    if ret != 0:
        error_message = "Failed to create Stream, ret=%s" % str(ret)
        print(error_message)
        exit()

    return stream_pi


def get_gallery_feature(input_dir, output_dir, stream_api):
    """
    Extract the features of gallery images, save the feature vector and the Pids to files

    :arg:
        imgPath: the directory of gallery images
        outputDir: the directory of gallery output files
        streamApi: stream api

    :return:
        None
    """
    
    # constructing the results returned by the queryImageProcess stream
    key_vec = StringVector()
    key_vec.push_back(b"mxpi_tensorinfer1")
        
    # check the query file
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
    features = []
    pids = []
    
    # extract the features for all images in gallery file
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.JPG'):
                data_input = MxDataInput()
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    data_input.data = f.read()
                start = time.time()
                # send the prepared data to the stream
                unique_id = stream_api.SendData(GALLERY_STREAM_NAME, IN_PLUGIN_ID, data_input)
                if unique_id < 0:
                    error_message = 'Failed to send data to queryImageProcess stream.'
                    print(error_message)
                    exit()
                # get infer result
                infer_result = stream_api.GetProtobuf(GALLERY_STREAM_NAME, OUT_PLUGIN_ID, key_vec)
                end = time.time()
                print("time:", end-start)
                # checking whether the infer results is valid or not
                if infer_result.size() == 0:
                    error_message = 'Unable to get effective infer results, please check the stream log for details'
                    print(error_message)
                    exit()
                                
                tensor_packages = MxpiDataType.MxpiTensorPackageList()
                tensor_packages.ParseFromString(infer_result[0].messageBuf)
                feature = np.frombuffer(tensor_packages.tensorPackageVec[0].tensorVec[0].dataStr,
                                                  dtype=np.float32)
                features.append(feature)
                pids.append(file.split('.')[0])
            else:
                print('Input image only support jpg')
                exit()
                
    features = np.array(features)
    features.tofile(os.path.join(output_dir, 'gallery_features.bin'))
    pids = np.array(pids).T
    np.savetxt(os.path.join(output_dir, 'persons.txt'), pids, fmt='%s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--galleryInputDir', type=str, default='data/gallery', help="Gallery File Path")
    parser.add_argument('--galleryOutputDir', type=str, default='output/gallery', help="Gallery Features Output Path")
    opt = parser.parse_args()
    stream_manager_api = initialize_stream()
    get_gallery_feature(opt.galleryInputDir, opt.galleryOutputDir, stream_manager_api)
    
    stream_manager_api.DestroyAllStreams()