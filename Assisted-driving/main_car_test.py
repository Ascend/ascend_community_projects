"""
Copyright 2022 Huawei Technologies Co., Ltd

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
import os
import stat
import argparse
import shutil
import cv2
from StreamManagerApi import StreamManagerApi, MxDataInput
from plot_utils import get_color_table, plot_one_box
check_ = ['car', 'bus', 'truck']
FLAGS = os.O_WRONLY | os.O_CREAT
MOD = stat.S_IWUSR | stat.S_IRUSR


def read_file_list(input_file):

    image_file_list = []
    with open(input_file, "r") as fs:
        for line in fs.readlines():
            line = line.strip('\n').split(' ')[1]
            image_file_list.append(line)
    return image_file_list


def save_infer_result(result_dir, result):

    load_dict = json.loads(result)
    if load_dict.get('MxpiObject') is None:
        with os.fdopen(os.open(result_dir + '/result.txt', FLAGS, MOD), 'a+') as f_write:
            object_list = 'Object detected num is 0\n'
            f_write.writelines(object_list)
    else:
        res_vec = load_dict.get('MxpiObject')
        with os.fdopen(os.open(result_dir + '/result.txt', FLAGS, MOD), 'a+') as f_write:
            object_list = 'Object detected num is ' + str(len(res_vec)) + '\n'
            f_write.writelines(object_list)
            for index, object_item in enumerate(res_vec):
                class_info = object_item.get('classVec')[0]
                object_info = '#Obj: ' + str(index) + ', box: ' + \
                    str(object_item.get('x0')) + ' ' + \
                    str(object_item.get('y0')) + ' ' + \
                    str(object_item.get('x1')) + ' ' + \
                    str(object_item.get('y1')) + ', confidence: ' + \
                    str(class_info.get('confidence')) + ', lable: ' + \
                    class_info.get('className') + ', id: ' + \
                    str(class_info.get('classId')) + '\n'
                f_write.writelines(object_info)


if __name__ == '__main__':
    # init stream manager

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer_file",
        type=str,
        default='./config/coco2014_minival.txt',
        help="path to dataset")
    parser.add_argument(
        "--res_dir_name",
        type=str,
        default='./data/car/result',
        help="path to dataset")
    parser.add_argument(
        "--pipeline",
        type=str,
        default='./pipeline/yolov3_opencv.pipeline',
        help="path to dataset")

    opt = parser.parse_args()
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(opt.pipeline, 'rb') as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    STREAMNAME = b'im_yolov3'
    INPLUGINLD = 0
    file_list = read_file_list(opt.infer_file)
    if len(file_list) == 0:
        print("the image folder is null")
        exit()
    color_table = get_color_table(80)
    try:
        shutil.rmtree(opt.res_dir_name)
    except BaseException:
        pass
    if not os.path.exists(opt.res_dir_name):
        os.makedirs(opt.res_dir_name)
    for file_path in file_list:
            file_name = os.path.join('data/car', file_path)
            print(file_name)
            data_input = MxDataInput()
            with open(file_name, 'rb') as f:
                data_input.data = f.read()

            unique_id = stream_manager.SendData(
                STREAMNAME, INPLUGINLD, data_input)

            if unique_id < 0:
                print("Failed to send data to stream.")
                exit()
            infer_result = stream_manager.GetResult(STREAMNAME, unique_id)
            if infer_result.errorCode != 0:
                print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                    infer_result.errorCode, infer_result.data.decode()))
                exit()
            save_infer_result(opt.res_dir_name, infer_result.data.decode())
    
    # destroy streams
    stream_manager.DestroyAllStreams()
