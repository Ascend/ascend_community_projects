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
import stat
import cv2

import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, MxProtobufIn, InProtobufVector, StringVector

from preprocess import preproc as preprocess
from visualize import plot_one_box

if __name__ == '__main__':
    steam_manager_api = StreamManagerApi()
    # init stream manager
    ret = steam_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    input_shape = [640, 640]
    # create streams by pipeline config file
    MODES = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open("../pipeline/nopre_post.pipeline", os.O_RDONLY, MODES), 'rb') as f:
        pipeline_str = f.read()
    ret = steam_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    # It is best to use absolute path
    IMG_PATH = "../test_img/test.jpg"
    RESULTFILE = "../test_img/nopre_post.jpg"
    if os.path.exists(IMG_PATH) != 1:
        print("The test image does not exist. Exit.")
        exit()
    origin_img = cv2.imread(IMG_PATH)

    lData, ratio = preprocess(origin_img, input_shape)
    STEAMNAME = b'detection'
    INPLUGINID = 0
    tensor = lData[None, :]
 
    visionList = MxpiDataType.MxpiVisionList()
    visionVec = visionList.visionVec.add()
    
    visionInfo = visionVec.visionInfo
    visionInfo.width = origin_img.shape[1]
    visionInfo.height = origin_img.shape[0]
    visionInfo.widthAligned = origin_img.shape[1]
    visionInfo.heightAligned = origin_img.shape[0]

    visionData = visionVec.visionData
    visionData.dataStr = tensor.tobytes()
    visionData.deviceId = 0
    visionData.memType = 0
    visionData.dataSize = len(tensor)
    
    KEY0 = b"appsrc0"

    protobufVec = InProtobufVector()

    
    protobuf = MxProtobufIn()
    protobuf.key = KEY0
    protobuf.type = b"MxTools.MxpiVisionList"
    protobuf.protobuf = visionList.SerializeToString()
    protobufVec.push_back(protobuf)
 
    uniqueId = steam_manager_api.SendProtobuf(STEAMNAME, INPLUGINID, protobufVec)


    if uniqueId < 0:
        print("Failed to send data to stream.")
        exit()
        
    key_plugins = [b"mxpi_objectpostprocessor0"]
    vector = StringVector()
    for key in key_plugins:
        vector.push_back(key)
    result = steam_manager_api.GetProtobuf(STEAMNAME, 0, vector)
    if result.size() == 0:
        print("No object detected")
        img = cv2.imread(IMG_PATH)
        cv2.imwrite(RESULTFILE, img)
        exit()
    if result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d, errorMsg=%s" % (
            result[0].errorCode, result[0].data.decode()))
        exit()
    # process data output from mxpi_objectpostprocessor plugin
    object_list = MxpiDataType.MxpiObjectList()
    object_list.ParseFromString(result[0].messageBuf)

    bounding_boxes = []
    for obj in object_list.objectVec:
        box = {'x0': int(obj.x0),
               'x1': int(obj.x1),
               'y0': int(obj.y0),
               'y1': int(obj.y1),
               'class': int(obj.classVec[0].classId),
               'class_name': obj.classVec[0].className,
               'confidence': round(obj.classVec[0].confidence, 4)}
        bounding_boxes.append(box)
    img = cv2.imread(IMG_PATH)
    # draw each bounding box on the original image
    for box in bounding_boxes:
        class_id = box.get('class')
        class_name = box.get('class_name')
        score = box.get('confidence')
        plot_one_box(img,
                     [box.get('x0'),
                      box.get('y0'),
                      box.get('x1'),
                      box.get('y1')],
                     cls_id=class_id,
                     label=class_name,
                     box_score=score)
    cv2.imwrite(RESULTFILE, img)
    # destroy streams
    steam_manager_api.DestroyAllStreams()
    

