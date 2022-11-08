#!/usr/bin/env python
# Copyright 2021 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import stat
import sys
import time
import argparse
from pprint import pprint
import rospy
from dr_spaam_ros import LaserDetROS, detections_to_pose_array, detections_to_rviz_marker
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker

from StreamManagerApi import StreamManagerApi, MxDataInput, StringVector
from StreamManagerApi import InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
import shutil
import matplotlib.pyplot as plt

from srcs.utils.utils import trim_the_scans
from srcs.utils.precision_recall import eval_internal

FLAGS = os.O_WRONLY | os.O_CREAT
MODES = stat.S_IWUSR | stat.S_IRUSR


def listener(ros_cls, output_save_dir=None):

    msg = rospy.wait_for_message("/segway/scan_multi", LaserScan, timeout=None)
    ros_cls.bag_id += 1
    scan = np.array(msg.ranges) # len of msg.ranges: 1091

    # added-in
    ros_cls.laser_scans.append(scan)  # append to the deque right
    anno_id = ros_cls.ts_frames[ros_cls.anno_id]['laser_frame']['url'].split('\\')[-1][:-4]
    frame_id = ros_cls.ts_frames[ros_cls.anno_id]["frame_id"] # detection saved as frame_id
    print("await for laser frame:", int(anno_id), "current bag id", ros_cls.bag_id, "frame id in timestamp", frame_id)
    while int(anno_id) == ros_cls.tested_id[-1]:
        # if the detection on frame_{anno_id} already exits
        txt_name = f"outputs/detections/{seq_name}/{str(frame_id).zfill(6)}.txt"
        dst_fname = os.path.join(output_save_dir, txt_name)
        src_fname = dst_fname[:-10] + ros_cls.ts_frames[ros_cls.anno_id-1]["frame_id"].zfill(6) + ".txt"
        shutil.copy(src_fname, dst_fname)
        ros_cls.anno_id += 1
        return
    if ros_cls.bag_id < int(anno_id):
        ros_cls.laser_scans.popleft()     # pop out from the deque left
        return
    else:
        ros_cls.anno_id += 1

    ros_cls.laser_scans.popleft()     # pop out from the deque left
    ros_cls.tested_id.append(int(anno_id))
    ros_cls.tested_id.popleft()

    if ros_cls.num_scans > 1:
        laser_scans = list(filter(lambda x: x is not None, ros_cls.laser_scans))
    else:
        laser_scans = ros_cls.laser_scans
    scan_index = len(laser_scans) - (ros_cls.bag_id - ros_cls.tested_id[-1])
    delta_inds = (np.arange(1, ros_cls.num_scans + 1) * ros_cls.stride)[::-1]
    scans_inds = [max(0, scan_index - i * ros_cls.stride) for i in delta_inds]
    scans = np.array([laser_scans[i] for i in scans_inds])

    # equivalent of flipping y axis (inversing laser phi angle)
    scans = scans[:, ::-1]

    laser_input = trim_the_scans(
            scans,
            ros_cls.scan_phi,
            stride=ros_cls.stride,
            **ros_cls.ct_kwargs,
        )

    t = time.time()
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    tsor = np.expand_dims(laser_input, axis=0).astype('<f4')
    array_bytes = tsor.tobytes()
    data_input = MxDataInput()
    data_input.data = array_bytes
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tsor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = data_input.data
    tensor_vec.tensorDataSize = len(array_bytes)

    key = "appsrc{}".format(ros_cls.in_plugin_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret = ros_cls.stream_manager_api.SendProtobuf(
        ros_cls.stream_name, ros_cls.in_plugin_id, protobuf_vec)

    if ret != 0:
        print("Failed to send data to stream.")
        exit()

    key_vec = StringVector()
    key_vec.push_back(b'mxpi_tensorinfer0')
    infer_result = ros_cls.stream_manager_api.GetProtobuf(ros_cls.stream_name, 0, key_vec)

    if infer_result.size() == 0:
        print("infer_result is null")
        exit()
    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
        exit()
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    pred_cls = np.frombuffer(
            result.tensorPackageVec[0].tensorVec[0].dataStr,
            dtype='<f4').reshape(
            tuple(
                result.tensorPackageVec[0].tensorVec[0].tensorShape))
    pred_reg = np.frombuffer(
            result.tensorPackageVec[0].tensorVec[1].dataStr,
            dtype='<f4').reshape(
            tuple(
                result.tensorPackageVec[0].tensorVec[1].tensorShape))
    prediction_shape = result.tensorPackageVec[0].tensorVec[1].tensorShape

    pred_cls_sigmoid = ros_cls.sigmoid(pred_cls.squeeze())
    dets_xy, dets_cls, inst_mask = ros_cls.nms(scans[-1], ros_cls.scan_phi, pred_cls_sigmoid, pred_reg.squeeze())
    print("[DrSpaamROS] End-to-end inference time: %f" % (t - time.time()))

    # dirty fix: save dets to file as roslaunch won't automatively terminate
    if dets_cls is None:
        dets_cls = np.ones(len(dets_xy), dtype=np.float32)
    # occluded for gts only
    occluded = np.zeros(len(dets_xy), dtype=np.int32)
    long_str = ""
    for category, xy, occ in zip(dets_cls, dets_xy, occluded):
        long_str += f"Pedestrian 0 {occ} 0 0 0 0 0 0 0 0 0 {xy[0]} {xy[1]} 0 0 {category}\n"
    long_str = long_str.strip("\n")
    txt_name = f"outputs/detections/{ros_cls.seq_name}/{str(frame_id).zfill(6)}.txt"
    det_fname = os.path.join(output_save_dir, txt_name)
    os.makedirs(os.path.dirname(det_fname), exist_ok=True)
    with os.fdopen(os.open(det_fname, FLAGS, MODES), "w") as fdo:
        fdo.write(long_str)

    # convert to ros msg and publish
    dets_msg = detections_to_pose_array(dets_xy, dets_cls)
    dets_msg.header = msg.header
    ros_cls.dets_pub.publish(dets_msg)

    rviz_msg = detections_to_rviz_marker(dets_xy, dets_cls)
    rviz_msg.header = msg.header
    ros_cls.rviz_pub.publish(rviz_msg)

    # dirty fix: no rendering support ONBOARD !!!
    if ros_cls.visu is False:
        print(len(dets_msg.poses), len(rviz_msg.points))
        fig, ax = ros_cls.plot_one_frame_beta(scan,
                                          ros_cls.scan_phi,
                                          ros_cls.bag_id,
                                          pred_reg.squeeze(),
                                          dets_msg.poses,
                                          rviz_msg.points,
                                        )
        fig_name = f"bags2png/{ros_cls.seq_name}/{str(ros_cls.bag_id).zfill(6)}.png"
        fig_file = os.path.join(output_save_dir, fig_name)
        print("Saving to {}...".format(fig_file))
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        fig.savefig(fig_file)
        plt.close(fig)


def echo(ros_cls, output_save_dir=None):

    anno_id = ros_cls.ts_frames[ros_cls.anno_id]['laser_frame']['url'].split('\\')[-1][:-4]
    frame_id = ros_cls.ts_frames[ros_cls.anno_id]["frame_id"]

    while int(anno_id) == ros_cls.tested_id[-1]:
        # if the detection on frame_{anno_id} already exits
        txt_name = f"outputs/detections/{seq_name}/{str(frame_id).zfill(6)}.txt"
        dst_fname = os.path.join(output_save_dir, txt_name)
        src_fname = dst_fname[:-10] + ros_cls.ts_frames[ros_cls.anno_id-1]["frame_id"].zfill(6) + ".txt"
        shutil.copy(src_fname, dst_fname)
        ros_cls.anno_id += 1
        return
    if ros_cls.bag_id < int(anno_id):
        ros_cls.laser_scans.popleft()     # pop out from the left
        return
    else:
        ros_cls.anno_id += 1

    ros_cls.laser_scans.popleft()     # pop out from the left
    ros_cls.tested_id.append(int(anno_id))
    ros_cls.tested_id.popleft()

    if ros_cls.num_scans > 1:
        laser_scans = list(filter(lambda x: x is not None, ros_cls.laser_scans))
    else:
        laser_scans = ros_cls.laser_scans
    scan_index = len(laser_scans) - (ros_cls.bag_id - ros_cls.tested_id[-1])
    delta_inds = (np.arange(1, ros_cls.num_scans + 1) * ros_cls.stride)[::-1]
    scans_inds = [max(0, scan_index - i * ros_cls.stride) for i in delta_inds]
    scans = np.array([laser_scans[i] for i in scans_inds])
    scans = scans[:, ::-1]

    laser_input = trim_the_scans(
            scans,
            ros_cls.scan_phi,
            stride=ros_cls.stride,
            **ros_cls.ct_kwargs,
        )

    t = time.time()
    tensor_package_list = MxpiDataType.MxpiTensorPackageList()
    tensor_package = tensor_package_list.tensorPackageVec.add()
    tsor = np.expand_dims(laser_input, axis=0).astype('<f4')
    array_bytes = tsor.tobytes()
    data_input = MxDataInput()
    data_input.data = array_bytes
    tensor_vec = tensor_package.tensorVec.add()
    tensor_vec.deviceId = 0
    tensor_vec.memType = 0
    for i in tsor.shape:
        tensor_vec.tensorShape.append(i)
    tensor_vec.dataStr = data_input.data
    tensor_vec.tensorDataSize = len(array_bytes)

    key = "appsrc{}".format(ros_cls.in_plugin_id).encode('utf-8')
    protobuf_vec = InProtobufVector()
    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b'MxTools.MxpiTensorPackageList'
    protobuf.protobuf = tensor_package_list.SerializeToString()
    protobuf_vec.push_back(protobuf)

    ret = ros_cls.stream_manager_api.SendProtobuf(ros_cls.stream_name, ros_cls.in_plugin_id, protobuf_vec)

    if ret != 0:
        print("Failed to send data to stream.")
        exit()

    key_vec = StringVector()
    key_vec.push_back(b'mxpi_tensorinfer0')
    infer_result = ros_cls.stream_manager_api.GetProtobuf(ros_cls.stream_name, 0, key_vec)

    if infer_result.size() == 0:
        print("infer_result is null")
        exit()
    if infer_result[0].errorCode != 0:
        print("GetProtobuf error. errorCode=%d" % (
                infer_result[0].errorCode))
        exit()
    result = MxpiDataType.MxpiTensorPackageList()
    result.ParseFromString(infer_result[0].messageBuf)
    pred_cls = np.frombuffer(
            result.tensorPackageVec[0].tensorVec[0].dataStr,
            dtype='<f4').reshape(
            tuple(
                result.tensorPackageVec[0].tensorVec[0].tensorShape))
    pred_reg = np.frombuffer(
            result.tensorPackageVec[0].tensorVec[1].dataStr,
            dtype='<f4').reshape(
            tuple(
                result.tensorPackageVec[0].tensorVec[1].tensorShape))
    prediction_shape = result.tensorPackageVec[0].tensorVec[1].tensorShape

    pred_cls_sigmoid = ros_cls.sigmoid(pred_cls.squeeze())
    dets_xy, dets_cls, inst_mask = ros_cls.nms(scans[-1], ros_cls.scan_phi, pred_cls_sigmoid, pred_reg.squeeze())
    print("[DrSpaamROS] End-to-end inference time: %f" % (t - time.time()))

    # dirty fix: save dets to file as roslaunch won't automatively terminate
    if dets_cls is None:
        dets_cls = np.ones(len(dets_xy), dtype=np.float32)
    # occluded for gts only
    occluded = np.zeros(len(dets_xy), dtype=np.int32)
    long_str = ""
    for category, xy, occ in zip(dets_cls, dets_xy, occluded):
        long_str += f"Pedestrian 0 {occ} 0 0 0 0 0 0 0 0 0 {xy[0]} {xy[1]} 0 0 {category}\n"
    long_str = long_str.strip("\n")
    txt_name = f"outputs/detections/{ros_cls.seq_name}/{str(frame_id).zfill(6)}.txt"
    det_fname = os.path.join(output_save_dir, txt_name)
    os.makedirs(os.path.dirname(det_fname), exist_ok=True)
    with os.fdopen(os.open(det_fname, FLAGS, MODES), "w") as fdo:
        fdo.write(long_str)


    # dirty fix: no rendering support ONBOARD !!!
    if ros_cls.visu is False:
        print(len(dets_msg.poses), len(rviz_msg.points))
        fig, ax = ros_cls.plot_one_frame_beta(scan,
                                          ros_cls.scan_phi,
                                          ros_cls.bag_id,
                                          pred_reg.squeeze(),
                                          dets_msg.poses,
                                          rviz_msg.points,
                                        )
        fig_name = f"bags2png/{ros_cls.seq_name}/{str(ros_cls.bag_id).zfill(6)}.png"
        fig_file = os.path.join(output_save_dir, fig_name)
        print("Saving to {}...".format(fig_file))
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        fig.savefig(fig_file)
        plt.close(fig)


if __name__ == '__main__':
    # init ros node here
    rospy.init_node("dr_spaam_ros")
    MODE_CHOOSE = 1
    SEQ_NAME = "bytes-cafe-2019-02-07_0"
    TIMESTAMPS_PATH = "frames_pc_im_laser.json"
    PIPE_STORE = "dr_spaam_jrdb_e20.pipeline"

    IS_RIVZ_SUPPORTED = False 
    # setup callback
    if MODE_CHOOSE == 2:
        try:
            LaserDetROS(PIPE_STORE, TIMESTAMPS_PATH, MODE_CHOOSE)
            print("** Node Launched **")
        except rospy.ROSInterruptException:
            pass
        rospy.spin()
    elif MODE_CHOOSE == 1:
        ROS_CLS = LaserDetROS(PIPE_STORE, TIMESTAMPS_PATH, MODE_CHOOSE)
        ROS_CLS.seq_name = SEQ_NAME
        ROS_CLS.visu = IS_RIVZ_SUPPORTED
        ROS_CLS.dets_pub = rospy.Publisher(
            "/laser_det_detections", PoseArray, queue_size=1, latch=False
        )

        ROS_CLS.rviz_pub = rospy.Publisher(
            "/laser_det_rviz", Marker, queue_size=1, latch=False
        )
        Output_save_dir = os.path.realpath(
                                os.path.join(os.getcwd(), os.path.dirname(__file__)))
        listener_loop = int(ROS_CLS.ts_frames[-1]['laser_frame']['url'].split('\\')[-1][:-4])
        while listener_loop >= 0:
            listener_loop -= 1
            print("loop", listener_loop)
            listener(ROS_CLS, Output_save_dir)

        while ROS_CLS.anno_id < len(ROS_CLS.ts_frames):
            print("apre", ROS_CLS.anno_id)
            echo(ROS_CLS, Output_save_dir)

        print("Mission completed, please check your output dir and welcome for the next use.")




