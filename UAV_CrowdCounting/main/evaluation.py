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
import os
import time
import xml.etree.ElementTree as ET
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector

class Dataset:
    '''
        Count the number of people for each picture in the test set.
            Args:
                the the test set path,
                the picture count file
            Returns:
                the test image name, datainput, keypoints
    '''
    def __init__(self, image_root_path, image_format='.jpg'):
        self.image_root_path = image_root_path
        self.image_format = image_format
        with open("./data/visdrone_test.txt") as txt:
            test_images = txt.read().split('\n')[:-1]
        self.images = []
        for image in test_images:
            self.images.append(os.path.join(image_root_path, "RGB", image + image_format))
        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_file = self.images[item]
        image_name = image_file.split("/")[-1]
        key_points = self.get_keypoints(image_file.replace("RGB", "GT_").replace(self.image_format, "R.xml"))
        image = self.default_loader(image_file)
        return image_name, image, len(key_points)

    @classmethod
    def default_loader(self, path):
        data_input = MxDataInput()
        with open(path, 'rb') as data:
            data_input.data = data.read()
        return data_input

    @classmethod
    def get_keypoints(self, xml):
        tree = ET.parse(xml)
        root = tree.getroot()
        key_points = root.findall("object")
        return key_points

if __name__ == '__main__':

    # Create a new StreamManager object and Initialize
    streamManagerApi = StreamManagerApi()
    ret = streamManagerApi.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # Building pipeline
    with open("./pipeline/crowdcount.pipeline", 'rb') as f:
        pipelineStr = f.read()

    ret = streamManagerApi.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    STREAM_NAME = b'uav_crowdcounting'
    IN_PLUGIN_ID = 0
    test_dataset = Dataset("./data/VisDrone2021/")
    count_errors = []
    t1 = time.time()
    for i, (name, dataInput, keypoints) in enumerate(test_dataset):
        uniqueId = streamManagerApi.SendData(STREAM_NAME, IN_PLUGIN_ID, dataInput)
        if uniqueId < 0:
            print("Failed to send data to stream.")
            exit()
        keys = [b"mxpi_tensorinfer0"]
        key_vec = StringVector()
        for key in keys:
            key_vec.push_back(key)
        infer_raw = streamManagerApi.GetResult(STREAM_NAME, b'appsink0', key_vec)
        infer_result = infer_raw.metadataVec[0]
        result = MxpiDataType.MxpiTensorPackageList()
        result.ParseFromString(infer_result.serializedMetadata)
        result_list = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr
                                    , dtype=np.float32)
        count_errors.append(keypoints - np.sum(result_list))
        print(name, keypoints, np.sum(result_list), (i + 1) / (time.time() - t1))

    fps = len(test_dataset) / (time.time() - t1)
    image_errs = np.array(count_errors)
    mse = np.sqrt(np.mean(np.square(image_errs)))
    mae = np.mean(np.abs(image_errs))
    print("Test. MSE: {}, MAE: {}, FPS: {}".format(mse, mae, fps))