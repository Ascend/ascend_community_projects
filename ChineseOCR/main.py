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

import glob
import os
import MxpiDataType_pb2 as MxpiDataType
import textdistance
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

FILE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.JPG', '*.PNG', '*.png', '*.JPEG']
LABEL_EXTENSIONS = ['*.TXT', '*.txt']
DATA_PATH = "output"

if __name__ == '__main__':
    # init stream manager
    PIPELINE_PATH = "chineseocr.pipeline"
    STREAMNAME = b'chineseocr'

    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open(PIPELINE_PATH, 'rb') as f:
        pipelineStr = f.read()
    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    INPLUGIN_ID = 0
    data_input = MxDataInput()
    output_path = os.path.dirname(os.path.realpath(__file__))
    output_exer = os.path.join(output_path, 'output')
    if not os.path.exists(output_exer):
        os.makedirs(output_exer)

    paths = []
    for extension in FILE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join("dataset", extension)))
    paths.sort()
    if len(paths) == 0:
        print("The dataset is empty!Only jpg or png format support.Please check the dataset and files.")
        exit()
    paths_label = []
    for extension in LABEL_EXTENSIONS:
        paths_label.extend(glob.glob(os.path.join("dataset", extension)))
    paths_label.sort()
    if len(paths_label) == 0:
        print("The label is none! We need input the txt and image file together.Please check the dataset and files.")
        exit()

    SCORE = 0
    NUM = 0
    for index, img_path in enumerate(paths):
        text_label = img_path.replace('jpg', 'txt')
        with open(img_path, 'rb') as fp:
            data_input.data = fp.read()
        unique_id = stream_manager_api.SendData(STREAMNAME, b'appsrc0', data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        key_vec = StringVector()
        key_vec.push_back(b'mxpi_textgenerationpostprocessor0')
        infer_result = stream_manager_api.GetProtobuf(STREAMNAME, INPLUGIN_ID, key_vec)

        if infer_result.size() == 0:
            print("infer_result is null")
            exit()
        if infer_result[0].errorCode != 0:
            print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
            exit()

        result = MxpiDataType.MxpiTextsInfoList()
        result.ParseFromString(infer_result[0].messageBuf)
        CONTENT_PIC = str(result.textsInfoVec[0].text)
        print(CONTENT_PIC[2:-2])
        NUM += 1

        with open(text_label, 'r', encoding='utf-8') as f:
            CONTENT = ""
            for i in f.readlines():
                CONTENT += i.strip()

            wd = os.open(os.path.join(output_exer, f'{index}.txt'), os.O_RDWR | os.O_CREAT, 0o660)
            wf = os.fdopen(wd, 'w', 0o660)
            wf.write(CONTENT)
            wf.close()

            wd = os.open(os.path.join(output_exer, f'{index}_pic.txt'), os.O_RDWR | os.O_CREAT, 0o660)
            wf = os.fdopen(wd, 'w', 0o660)
            wf.write(CONTENT_PIC[2:-2])
            wf.close()

        SCORE += textdistance.hamming.normalized_similarity(CONTENT_PIC[2:-2], CONTENT)

    print("the accuracy is:", SCORE / NUM)
    # destroy streams
    stream_manager_api.DestroyAllStreams()
