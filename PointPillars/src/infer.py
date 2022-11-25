"""
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
"""
import math
import datetime
import fire
import numpy as np
import MxpiDataType_pb2 as MxpiDataType
from StreamManagerApi import StreamManagerApi, StringVector, MxDataInput, InProtobufVector, MxProtobufIn


def get_pseudo_image(pillar_feature, coors):
    pseudo_image = np.zeros((1, 64, 496, 432))
    for i in range(0, 12000):
        x = math.ceil(coors[i, 0])
        y = math.ceil(coors[i, 1])
        for j in range(0, 64):
            pseudo_image[0, j, y, x] = pillar_feature[0, j, i, 0]
    return pseudo_image


def infer(file_dir = "../data/test/"):
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    path = b"../pipeline/pfe.pipeline"
    ret = stream_manager_api.CreateMultipleStreamsFromFile(path)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    stream_name = b'pfe'
    
    stream_manager_api_rpn = StreamManagerApi()
    ret = stream_manager_api_rpn.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    path_rpn = b"../pipeline/rpn.pipeline"
    ret = stream_manager_api_rpn.CreateMultipleStreamsFromFile(path_rpn)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    stream_name_rpn = b'rpn'

    # Get the pillar_x 
    pillar_x = np.fromfile(f"{file_dir}/pillar_x.bin", dtype=np.float16)
    pillar_x = pillar_x.astype(np.float16).reshape((1, 12000, 100))
    pillar_x_tensor = pillar_x[None]
    print("---------------PILLAR_X INFO--------------")
    print(pillar_x_tensor.size)
    print(pillar_x_tensor.shape)
    pillar_x_panckage_list = MxpiDataType.MxpiTensorPackageList()
    pillar_x_panckage = pillar_x_panckage_list.tensorPackageVec.add()
    pillar_x_vec = pillar_x_panckage.tensorVec.add()

    pillar_x_byte = pillar_x_tensor.tobytes()
    pillar_x_input = MxDataInput()
    pillar_x_input.data = pillar_x_byte

    pillar_x_vec.deviceId = 0
    pillar_x_vec.memType = 0
    for i in pillar_x_tensor.shape:
        pillar_x_vec.tensorShape.append(i)
    pillar_x_vec.dataStr = pillar_x_input.data
    pillar_x_vec.tensorDataSize = len(pillar_x_byte)

    plugin_id_x = 0
    key = "appsrc{}".format(plugin_id_x).encode('utf-8')
    buffer_vec_x = InProtobufVector()
    xbuf = MxProtobufIn()
    xbuf.key = key
    xbuf.type = b'MxTools.MxpiTensorPackageList'
    xbuf.protobuf = pillar_x_panckage_list.SerializeToString()
    buffer_vec_x.push_back(xbuf)

    # Get the pillar_y
    pillar_y = np.fromfile(f"{file_dir}/pillar_y.bin", dtype=np.float16)
    pillar_y = pillar_y.astype(np.float16).reshape((1, 12000, 100))
    pillar_y_tensor = pillar_y[None]
    print("---------------PILLAR_Y INFO--------------")
    print(pillar_y_tensor.size)
    print(pillar_y_tensor.shape)
    pillar_y_panckage_list = MxpiDataType.MxpiTensorPackageList()
    pillar_y_panckage = pillar_y_panckage_list.tensorPackageVec.add()
    pillar_y_vec = pillar_y_panckage.tensorVec.add()

    pillar_y_byte = pillar_y_tensor.tobytes()
    pillar_y_input = MxDataInput()
    pillar_y_input.data = pillar_y_byte

    pillar_y_vec.deviceId = 0
    pillar_y_vec.memType = 0
    for i in pillar_y_tensor.shape:
        pillar_y_vec.tensorShape.append(i)
    pillar_y_vec.dataStr = pillar_y_input.data
    pillar_y_vec.tensorDataSize = len(pillar_y_byte)

    plugin_id_y = 1
    key = "appsrc{}".format(plugin_id_y).encode('utf-8')
    buffer_vec_y = InProtobufVector()
    ybuf = MxProtobufIn()
    ybuf.key = key
    ybuf.type = b'MxTools.MxpiTensorPackageList'
    ybuf.protobuf = pillar_y_panckage_list.SerializeToString()
    buffer_vec_y.push_back(ybuf)

    # Get the pillar_z
    pillar_z = np.fromfile(f"{file_dir}/pillar_z.bin", dtype=np.float16)
    pillar_z = pillar_z.astype(np.float16).reshape((1, 12000, 100))
    pillar_z_tensor = pillar_z[None]
    print("---------------PILLAR_Z INFO--------------")
    print(pillar_z_tensor.size)
    print(pillar_z_tensor.shape)
    pillar_z_panckage_list = MxpiDataType.MxpiTensorPackageList()
    pillar_z_panckage = pillar_z_panckage_list.tensorPackageVec.add()
    pillar_z_vec = pillar_z_panckage.tensorVec.add()

    pillar_z_byte = pillar_z_tensor.tobytes()
    pillar_z_input = MxDataInput()
    pillar_z_input.data = pillar_z_byte

    pillar_z_vec.deviceId = 0
    pillar_z_vec.memType = 0
    for i in pillar_z_tensor.shape:
        pillar_z_vec.tensorShape.append(i)
    pillar_z_vec.dataStr = pillar_z_input.data
    pillar_z_vec.tensorDataSize = len(pillar_z_byte)

    plugin_id_z = 2
    key = "appsrc{}".format(plugin_id_z).encode('utf-8')
    buffer_vec_z = InProtobufVector()
    zbuf = MxProtobufIn()
    zbuf.key = key
    zbuf.type = b'MxTools.MxpiTensorPackageList'
    zbuf.protobuf = pillar_z_panckage_list.SerializeToString()
    buffer_vec_z.push_back(zbuf)

    # Get the pillar_i
    pillar_i = np.fromfile(f"{file_dir}/pillar_i.bin", dtype=np.float16)
    pillar_i = pillar_i.astype(np.float16).reshape((1, 12000, 100))
    pillar_i_tensor = pillar_i[None]
    print("---------------PILLAR_I INFO--------------")
    print(pillar_i_tensor.size)
    print(pillar_i_tensor.shape)
    pillar_i_panckage_list = MxpiDataType.MxpiTensorPackageList()
    pillar_i_panckage = pillar_i_panckage_list.tensorPackageVec.add()
    pillar_i_vec = pillar_i_panckage.tensorVec.add()

    pillar_i_byte = pillar_i_tensor.tobytes()
    pillar_i_input = MxDataInput()
    pillar_i_input.data = pillar_i_byte

    pillar_i_vec.deviceId = 0
    pillar_i_vec.memType = 0
    for i in pillar_i_tensor.shape:
        pillar_i_vec.tensorShape.append(i)
    pillar_i_vec.dataStr = pillar_i_input.data
    pillar_i_vec.tensorDataSize = len(pillar_i_byte)

    plugin_id_i = 3
    key = "appsrc{}".format(plugin_id_i).encode('utf-8')
    buffer_vec_i = InProtobufVector()
    ibuf = MxProtobufIn()
    ibuf.key = key
    ibuf.type = b'MxTools.MxpiTensorPackageList'
    ibuf.protobuf = pillar_i_panckage_list.SerializeToString()
    buffer_vec_i.push_back(ibuf)

    # Get the num_points_per_pillar
    num_points_per_pillar = np.fromfile(f"{file_dir}/num_points_per_pillar.bin", dtype=np.float16)
    num_points_per_pillar = num_points_per_pillar.astype(np.float16).reshape((12000,))
    num_points_per_pillar_tensor = num_points_per_pillar[None]
    print("---------------NUM INFO--------------")
    print(num_points_per_pillar_tensor.size)
    print(num_points_per_pillar_tensor.shape)
    num_points_per_pillar_panckage_list = MxpiDataType.MxpiTensorPackageList()
    num_points_per_pillar_panckage = num_points_per_pillar_panckage_list.tensorPackageVec.add()
    num_points_per_pillar_vec = num_points_per_pillar_panckage.tensorVec.add()

    num_points_per_pillar_byte = num_points_per_pillar_tensor.tobytes()
    num_points_per_pillar_input = MxDataInput()
    num_points_per_pillar_input.data = num_points_per_pillar_byte

    num_points_per_pillar_vec.deviceId = 0
    num_points_per_pillar_vec.memType = 0
    for i in num_points_per_pillar_tensor.shape:
        num_points_per_pillar_vec.tensorShape.append(i)
    num_points_per_pillar_vec.dataStr = num_points_per_pillar_input.data
    num_points_per_pillar_vec.tensorDataSize = len(num_points_per_pillar_byte)

    plugin_id_num = 4
    key = "appsrc{}".format(plugin_id_num).encode('utf-8')
    buffer_vec_num = InProtobufVector()
    numbuf = MxProtobufIn()
    numbuf.key = key
    numbuf.type = b'MxTools.MxpiTensorPackageList'
    numbuf.protobuf = num_points_per_pillar_panckage_list.SerializeToString()
    buffer_vec_num.push_back(numbuf)

    # Get the x_sub
    x_sub = np.fromfile(f"{file_dir}/x_sub_shaped.bin", dtype=np.float16)
    x_sub = x_sub.astype(np.float16).reshape((1, 12000, 100))
    x_sub_tensor = x_sub[None]
    print("---------------X_SUB INFO--------------")
    print(x_sub_tensor.size)
    print(x_sub_tensor.shape)
    x_sub_panckage_list = MxpiDataType.MxpiTensorPackageList()
    x_sub_panckage = x_sub_panckage_list.tensorPackageVec.add()
    x_sub_vec = x_sub_panckage.tensorVec.add()

    x_sub_byte = x_sub_tensor.tobytes()
    x_sub_input = MxDataInput()
    x_sub_input.data = x_sub_byte

    x_sub_vec.deviceId = 0
    x_sub_vec.memType = 0
    for i in x_sub_tensor.shape:
        x_sub_vec.tensorShape.append(i)
    x_sub_vec.dataStr = x_sub_input.data
    x_sub_vec.tensorDataSize = len(x_sub_byte)

    plugin_id_x_sub = 5
    key = "appsrc{}".format(plugin_id_x_sub).encode('utf-8')
    buffer_vec_x_sub = InProtobufVector()
    x_sub_buf = MxProtobufIn()
    x_sub_buf.key = key
    x_sub_buf.type = b'MxTools.MxpiTensorPackageList'
    x_sub_buf.protobuf = x_sub_panckage_list.SerializeToString()
    buffer_vec_x_sub.push_back(x_sub_buf)

    # Get the y_sub
    y_sub = np.fromfile(f"{file_dir}/y_sub_shaped.bin", dtype=np.float16)
    y_sub = y_sub.astype(np.float16).reshape((1, 12000, 100))
    y_sub_tensor = y_sub[None]
    print("---------------Y_SUB INFO--------------")
    print(y_sub_tensor.size)
    print(y_sub_tensor.shape)
    y_sub_panckage_list = MxpiDataType.MxpiTensorPackageList()
    y_sub_panckage = y_sub_panckage_list.tensorPackageVec.add()
    y_sub_vec = y_sub_panckage.tensorVec.add()

    y_sub_byte = y_sub_tensor.tobytes()
    y_sub_input = MxDataInput()
    y_sub_input.data = y_sub_byte

    y_sub_vec.deviceId = 0
    y_sub_vec.memType = 0
    for i in y_sub_tensor.shape:
        y_sub_vec.tensorShape.append(i)
    y_sub_vec.dataStr = y_sub_input.data
    y_sub_vec.tensorDataSize = len(y_sub_byte)

    plugin_id_y_sub = 6
    key = "appsrc{}".format(plugin_id_y_sub).encode('utf-8')
    buffer_vec_y_sub = InProtobufVector()
    y_sub_buf = MxProtobufIn()
    y_sub_buf.key = key
    y_sub_buf.type = b'MxTools.MxpiTensorPackageList'
    y_sub_buf.protobuf = y_sub_panckage_list.SerializeToString()
    buffer_vec_y_sub.push_back(y_sub_buf)

    # Get the mask
    mask = np.fromfile(f"{file_dir}/mask.bin", dtype=np.float16)
    mask = mask.astype(np.float16).reshape((1, 12000, 100))
    mask_tensor = mask[None]
    print("---------------MASK INFO--------------")
    print(mask_tensor.size)
    print(mask_tensor.shape)
    mask_panckage_list = MxpiDataType.MxpiTensorPackageList()
    mask_panckage = mask_panckage_list.tensorPackageVec.add()
    mask_vec = mask_panckage.tensorVec.add()

    mask_byte = mask_tensor.tobytes()
    mask_input = MxDataInput()
    mask_input.data = mask_byte

    mask_vec.deviceId = 0
    mask_vec.memType = 0
    for i in mask_tensor.shape:
        mask_vec.tensorShape.append(i)
    mask_vec.dataStr = mask_input.data
    mask_vec.tensorDataSize = len(mask_byte)

    plugin_id_mask = 7
    key = "appsrc{}".format(plugin_id_mask).encode('utf-8')
    buffer_vec_mask = InProtobufVector()
    mask_buf = MxProtobufIn()
    mask_buf.key = key
    mask_buf.type = b'MxTools.MxpiTensorPackageList'
    mask_buf.protobuf = mask_panckage_list.SerializeToString()
    buffer_vec_mask.push_back(mask_buf)

    # Send data to the stream
    unique_id_x = stream_manager_api.SendProtobuf(stream_name, plugin_id_x, buffer_vec_x)
    unique_id_y = stream_manager_api.SendProtobuf(stream_name, plugin_id_y, buffer_vec_y)
    unique_id_z = stream_manager_api.SendProtobuf(stream_name, plugin_id_z, buffer_vec_z)
    unique_id_i = stream_manager_api.SendProtobuf(stream_name, plugin_id_i, buffer_vec_i)
    unique_id_num = stream_manager_api.SendProtobuf(stream_name, plugin_id_num, buffer_vec_num)
    unique_id_x_sub = stream_manager_api.SendProtobuf(stream_name, plugin_id_x_sub, buffer_vec_x_sub)
    unique_id_y_sub = stream_manager_api.SendProtobuf(stream_name, plugin_id_y_sub, buffer_vec_y_sub)
    unique_id_mask = stream_manager_api.SendProtobuf(stream_name, plugin_id_mask, buffer_vec_mask)
    begin_time = datetime.datetime.now()
    if unique_id_x < 0 or unique_id_y < 0 or unique_id_z < 0 or unique_id_i < 0 \
       or unique_id_num < 0 or unique_id_x_sub < 0 or unique_id_y_sub < 0 or unique_id_mask < 0:
        print("Failed to send data to stream.")
        exit()

    key_vec = StringVector()
    key_vec.push_back(b'mxpi_tensorinfer0')
    # get inference result
    get_result = stream_manager_api.GetResult(stream_name, b'appsink0', key_vec)
    spend_time = (datetime.datetime.now() - begin_time).total_seconds()
    if get_result.errorCode != 0:
        print("ERROR")
        exit()
    print("-----------Result---------------")
    print(get_result)

    infer_result = get_result.metadataVec[0]

    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result.serializedMetadata)
    result.tensorPackageVec[0].tensorVec[0].dataStr
    result_np = np.frombuffer(result.tensorPackageVec[0].tensorVec[0].dataStr, dtype = np.float32)
    result_np.tofile(f"{file_dir}/feature.bin")

    # Pillar Scatter
    pillar_feature = np.fromfile(f"{file_dir}/feature.bin", dtype=np.float32)
    pillar_feature = pillar_feature.astype(np.float16).reshape((1, 64, 12000, 1))
    print(pillar_feature.shape)
    coors = np.load(f"{file_dir}/coor.npy")
    print(coors.shape) 
    pseudo_image = get_pseudo_image(pillar_feature, coors).astype(np.float16)
    print(pseudo_image.shape)
    pseudo_image.tofile(f"{file_dir}/pseudo_image.bin")

    # Get the pseudo image
    pseudo_image = np.fromfile(f"{file_dir}/pseudo_image.bin", dtype=np.float16)
    pseudo_image = pseudo_image.astype(np.float32).reshape((64, 496, 432))
    pseudo_image_tensor = pseudo_image[None]
    print("---------------PSEUDO IMAGE INFO--------------")
    print(pseudo_image_tensor.size)
    print(pseudo_image_tensor.shape)
    pseudo_image_panckage_list = MxpiDataType.MxpiTensorPackageList()
    pseudo_image_panckage = pseudo_image_panckage_list.tensorPackageVec.add()
    pseudo_image_vec = pseudo_image_panckage.tensorVec.add()

    pseudo_image_byte = pseudo_image_tensor.tobytes()
    pseudo_image_input = MxDataInput()
    pseudo_image_input.data = pseudo_image_byte

    pseudo_image_vec.deviceId = 0
    pseudo_image_vec.memType = 0
    for i in pseudo_image_tensor.shape:
        pseudo_image_vec.tensorShape.append(i)
    pseudo_image_vec.dataStr = pseudo_image_input.data
    pseudo_image_vec.tensorDataSize = len(pseudo_image_byte)

    plugin_id_pseudo_image = 0
    key = "appsrc{}".format(plugin_id_pseudo_image).encode('utf-8')
    buffer_vec_pseudo_image = InProtobufVector()
    pseudo_image_buf = MxProtobufIn()
    pseudo_image_buf.key = key
    pseudo_image_buf.type = b'MxTools.MxpiTensorPackageList'
    pseudo_image_buf.protobuf = pseudo_image_panckage_list.SerializeToString()
    buffer_vec_pseudo_image.push_back(pseudo_image_buf)

    # Send data to the stream
    unique_id_pseudo_image = stream_manager_api_rpn.\
        SendProtobuf(stream_name_rpn, plugin_id_pseudo_image, buffer_vec_pseudo_image)
    begin_time = datetime.datetime.now()
    if unique_id_pseudo_image < 0:
        print("Failed to send data to stream.")
        exit()


    key_vec = StringVector()
    key_vec.push_back(b'mxpi_tensorinfer0')
    # get inference result
    get_result = stream_manager_api_rpn.GetResult(stream_name_rpn, b'appsink0', key_vec)
    spend_time += (datetime.datetime.now() - begin_time).total_seconds()
    if get_result.errorCode != 0:
        print("ERROR")
        exit()
    print("-----------Result---------------")
    infer_result = get_result.metadataVec[0]
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result.serializedMetadata)
    result_box = result.tensorPackageVec[0].tensorVec[0].dataStr
    result_cls = result.tensorPackageVec[0].tensorVec[1].dataStr
    result_dir = result.tensorPackageVec[0].tensorVec[2].dataStr
    result_shape0 = result.tensorPackageVec[0].tensorVec[0].tensorShape
    result_shape1 = result.tensorPackageVec[0].tensorVec[1].tensorShape
    result_shape2 = result.tensorPackageVec[0].tensorVec[2].tensorShape
    print(result_shape0)
    print(result_shape1)
    print(result_shape2)
    result_box_np = np.frombuffer(result_box, dtype = np.float32)
    result_cls_np = np.frombuffer(result_cls, dtype = np.float32)
    result_dir_np = np.frombuffer(result_dir, dtype = np.float32)
    result_dir = "../result/test/"
    result_box_np.tofile(f"{result_dir}/box.bin")
    result_cls_np.tofile(f"{result_dir}/cls.bin")
    result_dir_np.tofile(f"{result_dir}/dir.bin")
    print("The total time consumed for model inference is : ", spend_time, "s")

if __name__ == '__main__':
    fire.Fire()
