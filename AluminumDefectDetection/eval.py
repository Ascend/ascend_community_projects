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
import json
import stat
import glob
import cv2
from StreamManagerApi import StreamManagerApi, MxDataInput
import numpy as np
from utils import xyxy2xywh,is_legal

from plots import box_label, colors

names = ['non_conduct', 'abrasion_mark', 'corner_leak', 'orange_peel', 'leak', 'jet_flow', 'paint_bubble', 'pit',
         'motley', 'dirty_spot']

if __name__ == '__main__':
    MODES = stat.S_IWUSR | stat.S_IRUSR
    # init stream manager
    dict_classes = {
        "non_conduct": "0", "abrasion_mark": "1", "corner_leak": "2", "orange_peel": "3", "leak": "4", "jet_flow": "5",
        "paint_bubble": "6", "pit": "7", "motley": "8", "dirty_spot": "9"
    }

    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    # create streams by pipeline config file
    with open("./pipeline/AlDefectDetection.pipeline", 'rb') as f:
        pipelineStr = f.read()
    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    print("load pipline done!")

    TESTIMGS = 0
    # Input object of streams -- detection target
    ORI_IMG_PATH = "./test/data/test/"
    TXT_PATH = "./test/" + "test_out_txt/"
    DETECT_IMG_PATH = "./test/" + "img_detected/"
    PRE_IMG_PATH = "./test/" + "img_pre/"
    if not os.path.exists(TXT_PATH):
        os.makedirs(TXT_PATH)
    if not os.path.exists(DETECT_IMG_PATH):
        os.makedirs(DETECT_IMG_PATH)
    if not os.path.exists(PRE_IMG_PATH):
        os.makedirs(PRE_IMG_PATH)

    files_list = glob.glob(ORI_IMG_PATH + '/*.jpg')
    if len(files_list) == 0:
        print("The input directory is EMPTY!")
        print("Please place the picture in './test/data/test/' !")
        exit()

    for item in os.listdir(ORI_IMG_PATH):

        ori_img_path = os.path.join(ORI_IMG_PATH, item)

        is_legal(ori_img_path)

        ori_img = cv2.imread(ori_img_path)  # 读取图片
        h0, w0 = ori_img.shape[:2]
        r = 640 / max(h0, w0)  # ratio
        input_shape = (640, 640)

        print("file_path:", ori_img_path)
        img_name = item.split(".")[0]
        img_txt = TXT_PATH + img_name + ".txt"
        if os.path.exists(img_txt):
            os.remove(img_txt)

        # Construct the input of the stream
        dataInput = MxDataInput()
        with open(ori_img_path, 'rb') as f:
            dataInput.data = f.read()

        # Inputs data to a specified stream based on streamName.
        STREAMNAME = b'classification+detection'
        INPLUGINID = 0
        uniqueId = streamManagerApi.SendDataWithUniqueId(STREAMNAME, INPLUGINID, dataInput)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()

        # Obtain the inference result by specifying streamName and uniqueId.
        inferResult = streamManagerApi.GetResultWithUniqueId(STREAMNAME, uniqueId, 5000)
        if inferResult.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                inferResult.errorCode, inferResult.data.decode()))
            exit()

        results = json.loads(inferResult.data.decode())
        if not results:
            print("No object detected")
            with os.fdopen(os.open(img_txt, os.O_RDWR | os.O_CREAT, MODES), 'a+') as f:
                pass
            continue
        img = cv2.imread(ori_img_path, cv2.IMREAD_COLOR)
        gn = np.array(ori_img.shape)[[1, 0, 1, 0]]

        bboxes = []
        classVecs = []
        for info in results['MxpiObject']:
            bboxes.append([float(info['x0']), float(info['y0']), float(info['x1']), float(info['y1'])])
            classVecs.append(info["classVec"])
        for (xyxy, classVec) in zip(bboxes, classVecs):
            xyxy = np.array(xyxy)
            xywh = (xyxy2xywh(xyxy.reshape(1, 4)) / gn).reshape(-1).tolist()  # normalized xywh
            try:
                line = (
                    int(dict_classes[classVec[0]["className"]]), *xywh,
                    round(classVec[0]["confidence"], 6))  # label format
            except KeyError:
                print("No sunch key")

            with os.fdopen(os.open(img_txt, os.O_RDWR | os.O_CREAT, MODES), 'a+') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

            label = f'{classVec[0]["className"]} {classVec[0]["confidence"]:.4f}'
            save_img = box_label(ori_img, xyxy, label, color=colors[names.index(classVec[0]["className"])])

        cv2.imwrite(DETECT_IMG_PATH + 'result' + item, save_img)
        TESTIMGS += 1
        ######################################################################################
        # print the infer result
        print(inferResult.data.decode())

    # Mark image count
    print("Image count:%d" % TESTIMGS)

    # destroy streams
    streamManagerApi.DestroyAllStreams()
