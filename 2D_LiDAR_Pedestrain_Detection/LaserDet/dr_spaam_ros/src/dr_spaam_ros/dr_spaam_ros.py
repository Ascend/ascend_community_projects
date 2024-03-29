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
import json
import shutil
from pprint import pprint
from collections import deque
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker
from std_msgs.msg import Int16
from srcs.detector import scans_to_cutout
from srcs.utils.precision_recall import eval_internal

from StreamManagerApi import StreamManagerApi, MxDataInput, MxBufferInput, StringVector
from StreamManagerApi import InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType
import numpy as np
import matplotlib.pyplot as plt
import rospy

FLAGS = os.O_WRONLY | os.O_CREAT
MODES = stat.S_IWUSR | stat.S_IRUSR


class LaserDetROS:
    """ROS node to detect pedestrian using DROW3 or DR-SPAAM."""

    def __init__(self, pipe_store, timestamps_path, mode=1):

        self.visu = True # False Or True
        self.seq_name = "rendering"
        self.bag_id = -1
        self.anno_id = 0
        # Set scan params
        self.conf_thresh = 0.5
        self.stride = 1
        self.panoramic_scan = True
        self.detector_model = os.path.basename(pipe_store).split("_")[0]
        if self.detector_model == "drow3":
            self.num_scans = 1
        else:
            self.num_scans = 10

        self.ct_kwargs = {
            "win_width": 1.0,
            "win_depth": 0.5,
            "num_ct_pts": 56,
            "pad_val": 29.99,
        }

        if mode == 2:
            self._init()

        self.laser_scans = deque([None] * self.num_scans)
        self.tested_id = deque([0])

        if "jrdb" in pipe_store or "JRDB" in pipe_store:
            self._laser_fov_deg = 360
            bisec_fov_rad = 0.5 * np.deg2rad(self._laser_fov_deg)
            self.scan_phi = np.linspace(
            -bisec_fov_rad, bisec_fov_rad, 1091, dtype=np.float32
            )
        elif "drow" in pipe_store or "DROW" in pipe_store:
            self._laser_fov_deg = 225
            bisec_fov_rad = 0.5 * np.deg2rad(self._laser_fov_deg)
            self.scan_phi = np.linspace(
            -bisec_fov_rad, bisec_fov_rad, 450, dtype=np.float32
            )

        self._timestamps = timestamps_path
        with open(self._timestamps, "rb") as f:
            self.ts_frames = json.load(f)["data"]
        self._mode = mode # 1: eval, 2: display

        # The following belongs to the SDK Process
        self.stream_manager_api = StreamManagerApi()
        # init stream manager
        ret = self.stream_manager_api.InitManager()
        if ret != 0:
            print("Failed to init Stream manager, ret=%s" % str(ret))
            exit()
        else:
            print("-----------------创建流管理StreamManager并初始化-----------------")

        # create streams by pipeline config file
        with open(pipe_store, 'rb') as f:
            print("-----------------正在读取读取pipeline-----------------")
            pipeline_str = f.read()
            print("-----------------成功读取pipeline-----------------")

        _ret = self.stream_manager_api.CreateMultipleStreams(pipeline_str)

        # Print error message
        if _ret != 0:
            print(
                "-----------------Failed to create Stream, ret=%s-----------------" %
                str(_ret))
        else:
            print(
                "-----------------Create Stream Successfully, ret=%s-----------------" %
                str(_ret))
        # Stream name

        self.stream_name = b'detection0'

        self.in_plugin_id = 0
        tb_dict_list = []
        dets_xy_accum = {}
        dets_cls_accum = {}
        dets_inds_accum = {}
        gts_xy_accum = {}
        gts_inds_accum = {}
        fig_dict = {}


    def __len__(self):
        return len(self.laser_scans)


    @classmethod
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))


    @classmethod
    def rphi_xy_convertor(self, rdius, ang):
        return rdius * np.cos(ang), rdius * np.sin(ang)


    @classmethod
    def _can_glob_convertor(self, rdius, ang, dx, dy):
        tmp_y = rdius + dy
        tmp_phi = np.arctan2(dx, tmp_y)
        dets_phi = tmp_phi + ang
        dets_r = tmp_y / np.cos(tmp_phi)
        return dets_r, dets_phi


    def nms(
        self, scan_grid, phi_grid, pred_cls, pred_reg, min_dist=0.5
        ):
        assert len(pred_cls.shape) == 1

        pred_r, pred_phi = self._can_glob_convertor(
            scan_grid, phi_grid, pred_reg[:, 0], pred_reg[:, 1]
        )
        pred_xs, pred_ys = self.rphi_xy_convertor(pred_r, pred_phi)

        # sort prediction with descending confidence
        sort_inds = np.argsort(pred_cls)[::-1]

        pred_xs, pred_ys = pred_xs[sort_inds], pred_ys[sort_inds]
        pred_cls = pred_cls[sort_inds]

        # compute pair-wise distance
        num_pts = len(scan_grid)
        xdiff = pred_xs.reshape(num_pts, 1) - pred_xs.reshape(1, num_pts)
        ydiff = pred_ys.reshape(num_pts, 1) - pred_ys.reshape(1, num_pts)
        p_dist = np.sqrt(np.square(xdiff) + np.square(ydiff))

        # nms
        keep = np.ones(num_pts, dtype=np.bool_)
        instance_mask = np.zeros(num_pts, dtype=np.int32)
        instance_id = 1
        for i in range(num_pts):
            if not keep[i]:
                continue

            dup_inds = p_dist[i] < min_dist
            keep[dup_inds] = False
            keep[i] = True
            instance_mask[sort_inds[dup_inds]] = instance_id
            instance_id += 1

        det_xys = np.stack((pred_xs, pred_ys), axis=1)[keep]
        det_cls = pred_cls[keep]

        return det_xys, det_cls, instance_mask


    def plot_one_frame_beta(
        self,
        scan_r,
        scan_phi,
        frame_idx,
        pred_reg=None,
        dets_msg_poses=None,
        rviz_msg_points=None,
        xlim=(-7, 7),
        ylim=(-7, 7),
        ):
        fig_handle = plt.figure(figsize=(10, 10))
        ax_handle = fig_handle.add_subplot(111)
        ax_handle.grid()

        ax_handle.set_xlim(xlim[0], xlim[1])
        ax_handle.set_ylim(ylim[0], ylim[1])
        ax_handle.set_xlabel("x [m]")
        ax_handle.set_ylabel("y [m]")
        ax_handle.set_aspect("equal")
        ax_handle.set_title(f"{frame_idx}")

        # plot scan
        scan_x, scan_y = self.rphi_xy_convertor(scan_r, scan_phi)
        ax_handle.scatter(scan_x, scan_y, s=1, c="blue")

        # plot rviz
        for idx, pt in enumerate(rviz_msg_points):
            if idx % 2 == 0:
                start_pt = (pt.x, pt.y)
            else:
                end_pt = (pt.x, pt.y)
                ax_handle.plot(start_pt, end_pt, linewidth=0.25, c="orange")

        # plot regression
        pred_r, pred_phi = self._can_glob_convertor(
            scan_r, scan_phi, pred_reg[:, 0], pred_reg[:, 1]
        )
        pred_x, pred_y = self.rphi_xy_convertor(pred_r, pred_phi)
        ax_handle.scatter(pred_x, pred_y, s=1, c="red")

        # plot detection
        for idx, pos in enumerate(dets_msg_poses):
            ax_handle.scatter(pos.position.x, pos.position.y,
                         marker="x", s=40, c="black")
            ax_handle.scatter(pos.position.x, pos.position.y,
                        s=200, facecolors="none", edgecolors="black")

        return fig_handle, ax_handle


    def _scan_callback(self, msg):

        self.bag_id += 1

        output_save_dir = os.path.realpath(
                            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        scan = np.array(msg.ranges) # len of msg.ranges: 1091
        scan[scan == 0.0] = 29.99
        scan[np.isinf(scan)] = 29.99
        scan[np.isnan(scan)] = 29.99

        # added-in
        self.laser_scans.append(scan)  # append to the right
        self.laser_scans.popleft()     # pop out from the left

        if self.num_scans > 1:
            laser_scans = list(filter(lambda x: x is not None, self.laser_scans))
        else:
            laser_scans = self.laser_scans
        scan_index = len(laser_scans)
        delta_inds = (np.arange(self.num_scans) * self.stride)[::-1]
        scans_inds = [max(0, scan_index - i) for i in delta_inds]

        scans = np.array([laser_scans[i-1] for i in scans_inds])
        scans = scans[:, ::-1]
        laser_input = scans_to_cutout(
            scans,
            self.scan_phi,
            stride=self.stride,
            win_size=[1.0, 0.5],
            num_cutout_pts=56,
            padding_val=29.99,
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

        key = "appsrc{}".format(self.in_plugin_id).encode('utf-8')
        protobuf_vec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = key
        protobuf.type = b'MxTools.MxpiTensorPackageList'
        protobuf.protobuf = tensor_package_list.SerializeToString()
        protobuf_vec.push_back(protobuf)

        ret = self.stream_manager_api.SendProtobuf(self.stream_name, self.in_plugin_id, protobuf_vec)

        if ret != 0:
            print("Failed to send data to stream.")
            exit()

        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        infer_result = self.stream_manager_api.GetProtobuf(self.stream_name, 0, key_vec)

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

        pred_cls_sigmoid = self._sigmoid(pred_cls.squeeze())
        dets_xy, dets_cls, inst_mask = self.nms(scans[-1], self.scan_phi, pred_cls_sigmoid, pred_reg.squeeze())
        print("[DrSpaamROS] End-to-end inference time: %f" % (t - time.time()))

        # confidence threshold (for visulization ONLY)
        conf_mask = (dets_cls >= self.conf_thresh).reshape(-1)
        dets_xy = dets_xy[conf_mask]
        dets_cls = dets_cls[conf_mask]

        # convert to ros msg and publish
        dets_msg = detections_to_pose_array(dets_xy, dets_cls)
        dets_msg.header = msg.header
        self.dets_pub.publish(dets_msg)

        rviz_msg = detections_to_rviz_marker(dets_xy, dets_cls)
        rviz_msg.header = msg.header
        self.rviz_pub.publish(rviz_msg)

        # dirty fix: no rendering support ONBOARD !!!
        if self.visu is False:
            print(len(dets_msg.poses), len(rviz_msg.points))
            fig, ax = self.plot_one_frame_beta(scan,
                                          self.scan_phi,
                                          self.bag_id,
                                          pred_reg.squeeze(),
                                          dets_msg.poses,
                                          rviz_msg.points,
                                        )
            fig_name = f"bags2png/{self.seq_name}/{str(self.bag_id).zfill(6)}.png"
            fig_file = os.path.join(output_save_dir, fig_name)
            print("Saving to {}...".format(fig_file))
            os.makedirs(os.path.dirname(fig_file), exist_ok=True)
            fig.savefig(fig_file)
            plt.close(fig)


    def _init(self):
        """
        @brief      Initialize ROS connection.
        """
        # Publisher
        self.dets_pub = rospy.Publisher(
            "/laser_det_detections", PoseArray, queue_size=1, latch=False
        )
        self.rviz_pub = rospy.Publisher(
            "/laser_det_rviz", Marker, queue_size=1, latch=False
        )

        # Subscriber
        self._scan_sub = rospy.Subscriber(
            "/segway/scan_multi", LaserScan, self._scan_callback, queue_size=1
        )


def detections_to_rviz_marker(dets_xy, dets_cls):
    message = Marker()
    message.action = Marker.ADD
    message.ns = "dr_spaam_ros" # name_space
    message.id = 0
    message.type = Marker.LINE_LIST

    # set quaternion so that RViz does not give warning
    message.pose.orientation.x = 0.0
    message.pose.orientation.y = 0.0
    message.pose.orientation.z = 0.0
    message.pose.orientation.w = 1.0

    message.scale.x = 0.03  # line width
    # color in red
    message.color.r = 1.0
    message.color.a = 1.0

    # draw a circle
    r = 0.4
    ang = np.linspace(0, 2 * np.pi, 20)
    xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)

    # to msg
    for d_xy, d_cls in zip(dets_xy, dets_cls):
        for i in range(len(xy_offsets) - 1):
            # start point of a segment
            point_0 = Point()
            point_0.x = d_xy[0] + xy_offsets[i, 0]
            point_0.y = d_xy[1] + xy_offsets[i, 1]
            point_0.z = 0.0
            message.points.append(point_0)

            # end point
            point_1 = Point()
            point_1.x = d_xy[0] + xy_offsets[i + 1, 0]
            point_1.y = d_xy[1] + xy_offsets[i + 1, 1]
            point_1.z = 0.0
            message.points.append(point_1)

    return message


def detections_to_pose_array(dets_xy, dets_cls):
    pose_array_1 = PoseArray()
    for d_xy, d_cls in zip(dets_xy, dets_cls):
        # Detector uses following frame convention:
        # x forward, y rightward, z downward, phi is angle w.r.t. x-axis
        p_tmp = Pose()
        p_tmp.position.z = 0.0
        p_tmp.position.x = d_xy[0]
        p_tmp.position.y = d_xy[1]
        pose_array_1.poses.append(p_tmp)

    return pose_array_1


def read_subscriber_param(name_tmp):
    topic_tmp = rospy.get_param("~subscriber/" + name_tmp + "/topic")
    queue_size_tmp = rospy.get_param("~subscriber/" + name_tmp + "/queue_size")
    return topic_tmp, queue_size_tmp


def read_publisher_param(name_tmp):
    topic_tmp = rospy.get_param("~publisher/" + name_tmp + "/topic")
    queue_size_tmp = rospy.get_param("~publisher/" + name_tmp + "/queue_size")
    latch_tmp = rospy.get_param("~publisher/" + name_tmp + "/latch")
    return topic_tmp, queue_size_tmp, latch_tmp
