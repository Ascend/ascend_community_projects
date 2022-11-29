# -*-coding:utf-8-*-

# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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
import shutil
import argparse
import numpy as np
import cv2
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_folder",
        type=str,
        default="./data/road_line/data/validation/images",
        help="path to dataset")
    parser.add_argument(
        "--save_image",
        type=str,
        default="./data/road_line/pre_mask",
        help="path to dataset")
    opt = parser.parse_args()
    try:
        shutil.rmtree(opt.save_image)
    except BaseException:
        pass
    os.mkdir(opt.save_image)
    Image_ = [os.path.join(opt.image_folder, files)
              for files in os.listdir(opt.image_folder)]
    if len(Image_) == 0:
        print("the image is null")
        exit()
    StrManagApi = StreamManagerApi()
    ret_tmp = StrManagApi.InitManager()
    STREAMNAME = b'segmentation'
    IN_PLUGIN_ID = 0
    if ret_tmp != 0:
        exit()

    with open("pipeline/road.pipeline", 'rb') as f:
        pipeline = f.read().replace(b'\r', b'').replace(b'\n', b'')
    pip_str = pipeline

    ret_tmp = StrManagApi.CreateMultipleStreams(pip_str)
    if ret_tmp != 0:
        exit()

    Data_tmp_ipuut = MxDataInput()
    for k, file_ in enumerate(Image_):
        if os.path.exists(file_) != 1:
            continue
        if 'png' in file_.split('/')[-1]:
            print('png images are not supported')
            exit()
        print("processing  {} ".format(file_))
        with open(file_, 'rb') as f:
            Data_tmp_ipuut.data = f.read()

        unique_id = StrManagApi.SendData(
            STREAMNAME, IN_PLUGIN_ID, Data_tmp_ipuut)
        if unique_id < 0:
            print("Failed to send data to stream.")
            continue

        keys = [b"mxpi_tensorinfer0"]
        keyVec = StringVector()
        for name in keys:
            keyVec.push_back(name)

        infer_result_tmp = StrManagApi.GetProtobuf(STREAMNAME, 0, keyVec)

        if len(infer_result_tmp) == 0:
            continue
        if infer_result_tmp[0].errorCode != 0:
            continue

        tensor_list_tmp = MxpiDataType.MxpiTensorPackageList()
        tensor_list_tmp.ParseFromString(infer_result_tmp[0].messageBuf)

        result = np.frombuffer(
            tensor_list_tmp.tensorPackageVec[0].tensorVec[0].dataStr,
            dtype=np.float32)
        result_shape = tensor_list_tmp.tensorPackageVec[0].tensorVec[0].tensorShape
        result = np.reshape(result, result_shape)
        results = np.copy(result)
        results[results >= 0.5] = 255
        results[results < 0.5] = 0
        _ = cv2.imread(file_)
        mas_tmp = results.reshape(
            448,
            448,
            1)
        mas_tmp = cv2.cvtColor(mas_tmp, cv2.COLOR_GRAY2BGR)
        mas_tmp = cv2.resize(mas_tmp, (_.shape[1], _.shape[0]))
        save_ = os.path.join(opt.save_image, file_.split('/')[-1])
        tmp_img = cv2.addWeighted(_, 0.5, mas_tmp, 0.8, 0, dtype=cv2.CV_32F)
        cv2.imwrite(save_, tmp_img)
