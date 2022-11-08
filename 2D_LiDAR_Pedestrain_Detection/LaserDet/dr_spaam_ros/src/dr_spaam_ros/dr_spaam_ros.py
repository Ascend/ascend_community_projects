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
import time
import json
import numpy as np
import shutil
import matplotlib.pyplot as plt
from collections import deque
import rospy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker
from std_msgs.msg import Int16

import os
import stat
import sys
sys.path.append("/home/HwHiAiUser/edge_dev/2D_LiDAR_Pedestrain_Detection/LaserDet")
from pprint import pprint
pprint(sys.path)
from srcs.detector import scans_to_cutout
from srcs.utils.precision_recall import eval_internal

from StreamManagerApi import StreamManagerApi, MxDataInput, MxBufferInput, StringVector
from StreamManagerApi import InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType


FLAGS = os.O_WRONLY | os.O_CREAT
MODES = stat.S_IWUSR | stat.S_IRUSR


class LaserDetROS:
    """ROS node to detect pedestrian using DROW3 or DR-SPAAM."""

    def __init__(self, pipe_store, timestamps_path, mode=1):

        self.visu = True # False Or True
        self._seq_name = "rendering"
        self._bag_id = -1
        self._anno_id = 0
        # Set scan params
        self.conf_thresh = 0.5
        self.stride = 1
        self.panoramic_scan = True
        self.detector_model = os.path.basename(pipe_store).split("_")[0]
        if self.detector_model == "drow3":
            self._num_scans = 1
        else:
            self._num_scans = 10

        self.ct_kwargs = {
            "win_width": 1.0,
            "win_depth": 0.5,
            "num_ct_pts": 56,
            "pad_val": 29.99,
        }

        if mode == 2:
            self._init()

        self._laser_scans = deque([None] * self._num_scans)
        self._tested_id = deque([0])

        if "jrdb" in pipe_store or "JRDB" in pipe_store:
            self._laser_fov_deg = 360
            bisec_fov_rad = 0.5 * np.deg2rad(self._laser_fov_deg)
            self._scan_phi = np.linspace(
            -bisec_fov_rad, bisec_fov_rad, 1091, dtype=np.float32
            )
        elif "drow" in pipe_store or "DROW" in pipe_store:
            self._laser_fov_deg = 225
            bisec_fov_rad = 0.5 * np.deg2rad(self._laser_fov_deg)
            self._scan_phi = np.linspace(
            -bisec_fov_rad, bisec_fov_rad, 450, dtype=np.float32
            )

        self._timestamps = timestamps_path
        with open(self._timestamps, "rb") as f:
            self._ts_frames = json.load(f)["data"]
        self._mode = mode # 1: eval, 2: display

        # The following belongs to the SDK Process
        self._stream_manager_api = StreamManagerApi()
        # init stream manager
        ret = self._stream_manager_api.InitManager()
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

        _ret = self._stream_manager_api.CreateMultipleStreams(pipeline_str)

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

        self._stream_name = b'detection0'

        self.in_plugin_id = 0
        tb_dict_list = []
        dets_xy_accum = {}
        dets_cls_accum = {}
        dets_inds_accum = {}
        gts_xy_accum = {}
        gts_inds_accum = {}
        fig_dict = {}


    def _init(self):
        """
        @brief      Initialize ROS connection.
        """
        # Publisher
        #topic, queue_size, latch = read_publisher_param("detections")
        # /dr_spaam_detections, 1, False
        self._dets_pub = rospy.Publisher(
            "/laser_det_detections", PoseArray, queue_size=1, latch=False
        )

        #topic, queue_size, latch = read_publisher_param("rviz")
        # /dr_spaam_rviz, 1, False
        self._rviz_pub = rospy.Publisher(
            "/laser_det_rviz", Marker, queue_size=1, latch=False
        )

        # Subscriber
        #topic, queue_size = read_subscriber_param("scan")
        # /segway/scan_multi, 1

        self._scan_sub = rospy.Subscriber(
            "/segway/scan_multi", LaserScan, self._scan_callback, queue_size=1
        )


    def __len__(self):
        return len(self._laser_scans)


    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))


    def _rphi_xy_convertor(self, rdius, ang):
        return rdius * np.cos(ang), rdius * np.sin(ang)


    def _can_glob_convertor(self, rdius, ang, dx, dy):
        tmp_y = rdius + dy
        tmp_phi = np.arctan2(dx, tmp_y)
        dets_phi = tmp_phi + ang
        dets_r = tmp_y / np.cos(tmp_phi)
        return dets_r, dets_phi


    def _nms(
        self, scan_grid, phi_grid, pred_cls, pred_reg, min_dist=0.5
        ):
        assert len(pred_cls.shape) == 1

        pred_r, pred_phi = self._can_glob_convertor(
            scan_grid, phi_grid, pred_reg[:, 0], pred_reg[:, 1]
        )
        pred_xs, pred_ys = self._rphi_xy_convertor(pred_r, pred_phi)

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


    def _plot_one_frame_beta(
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
        scan_x, scan_y = self._rphi_xy_convertor(scan_r, scan_phi)
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
        pred_x, pred_y = self._rphi_xy_convertor(pred_r, pred_phi)
        ax_handle.scatter(pred_x, pred_y, s=1, c="red")

        # plot detection
        for idx, pos in enumerate(dets_msg_poses):
            ax_handle.scatter(pos.position.x, pos.position.y,
                         marker="x", s=40, c="black")
            ax_handle.scatter(pos.position.x, pos.position.y,
                        s=200, facecolors="none", edgecolors="black")

        return fig_handle, ax_handle


    def _scan_callback(self, msg):

        self._bag_id += 1

        output_save_dir = os.path.realpath(
                            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        scan = np.array(msg.ranges) # len of msg.ranges: 1091
        scan[scan == 0.0] = 29.99
        scan[np.isinf(scan)] = 29.99
        scan[np.isnan(scan)] = 29.99

        # added-in
        self._laser_scans.append(scan)  # append to the right
        self._laser_scans.popleft()     # pop out from the left

        if self._num_scans > 1:
            laser_scans = list(filter(lambda x: x is not None, self._laser_scans))
        else:
            laser_scans = self._laser_scans
        scan_index = len(laser_scans)
        delta_inds = (np.arange(self._num_scans) * self.stride)[::-1]
        scans_inds = [max(0, scan_index - i) for i in delta_inds]

        scans = np.array([laser_scans[i-1] for i in scans_inds])
        scans = scans[:, ::-1]
        laser_input = scans_to_cutout(
            scans,
            self._scan_phi,
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

        ret = self._stream_manager_api.SendProtobuf(self._stream_name, self.in_plugin_id, protobuf_vec)

        if ret != 0:
            print("Failed to send data to stream.")
            exit()

        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        infer_result = self._stream_manager_api.GetProtobuf(self._stream_name, 0, key_vec)

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
        dets_xy, dets_cls, inst_mask = self._nms(scans[-1], self._scan_phi, pred_cls_sigmoid, pred_reg.squeeze())
        print("[DrSpaamROS] End-to-end inference time: %f" % (t - time.time()))
        '''
        # dirty fix: save dets to file as roslaunch won't automatively terminate
        if dets_cls is None:
            dets_cls = np.ones(len(dets_xy), dtype=np.float32)
        # occluded for gts only
        occluded = np.zeros(len(dets_xy), dtype=np.int32)
        long_str = ""
        for cls, xy, occ in zip(dets_cls, dets_xy, occluded):
            long_str += f"Pedestrian 0 {occ} 0 0 0 0 0 0 0 0 0 {xy[0]} {xy[1]} 0 0 {cls}\n"
        long_str = long_str.strip("\n")
        txt_name = f"outputs/detections/{self._seq_name}/{str(frame_id).zfill(6)}.txt"
        det_fname = os.path.join(output_save_dir, txt_name)
        os.makedirs(os.path.dirname(det_fname), exist_ok=True)
        with os.fdopen(os.open(det_fname, FLAGS, MODES), "w") as fdo:
            fdo.write(long_str)
        '''
        # confidence threshold (for visulization ONLY)
        conf_mask = (dets_cls >= self.conf_thresh).reshape(-1)
        dets_xy = dets_xy[conf_mask]
        dets_cls = dets_cls[conf_mask]

        # convert to ros msg and publish
        dets_msg = detections_to_pose_array(dets_xy, dets_cls)
        dets_msg.header = msg.header
        self._dets_pub.publish(dets_msg)

        rviz_msg = detections_to_rviz_marker(dets_xy, dets_cls)
        rviz_msg.header = msg.header
        self._rviz_pub.publish(rviz_msg)

        # dirty fix: no rendering support ONBOARD !!!
        if self.visu == False:
            print(len(dets_msg.poses), len(rviz_msg.points))
            fig, ax = self._plot_one_frame_beta(scan,
                                          self._scan_phi,
                                          self._bag_id,
                                          pred_reg.squeeze(),
                                          dets_msg.poses,
                                          rviz_msg.points,
                                        )
            fig_name = f"bags2png/{self._seq_name}/{str(self._bag_id).zfill(6)}.png"
            fig_file = os.path.join(output_save_dir, fig_name)
            print("Saving to {}...".format(fig_file))
            os.makedirs(os.path.dirname(fig_file), exist_ok=True)
            fig.savefig(fig_file)
            plt.close(fig)



def detections_to_rviz_marker(dets_xy, dets_cls):
    """
    @brief     Convert detection to RViz marker msg. Each detection is marked as
               a circle approximated by line segments.
    """
    msg = Marker()
    msg.action = Marker.ADD
    msg.ns = "dr_spaam_ros" # name_space
    msg.id = 0
    msg.type = Marker.LINE_LIST

    # set quaternion so that RViz does not give warning
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = 1.0

    msg.scale.x = 0.03  # line width
    # red color
    msg.color.r = 1.0
    msg.color.a = 1.0

    # circle
    r = 0.4
    ang = np.linspace(0, 2 * np.pi, 20)
    xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)

    # to msg
    for d_xy, d_cls in zip(dets_xy, dets_cls):
        for i in range(len(xy_offsets) - 1):
            # start point of a segment
            p0 = Point()
            p0.x = d_xy[0] + xy_offsets[i, 0]
            p0.y = d_xy[1] + xy_offsets[i, 1]
            p0.z = 0.0
            msg.points.append(p0)

            # end point
            p1 = Point()
            p1.x = d_xy[0] + xy_offsets[i + 1, 0]
            p1.y = d_xy[1] + xy_offsets[i + 1, 1]
            p1.z = 0.0
            msg.points.append(p1)

    return msg


def detections_to_pose_array(dets_xy, dets_cls):
    pose_array = PoseArray()
    for d_xy, d_cls in zip(dets_xy, dets_cls):
        # Detector uses following frame convention:
        # x forward, y rightward, z downward, phi is angle w.r.t. x-axis
        p = Pose()
        p.position.x = d_xy[0]
        p.position.y = d_xy[1]
        p.position.z = 0.0
        pose_array.poses.append(p)

    return pose_array


def read_subscriber_param(name):
    """
    @brief      Convenience function to read subscriber parameter.
    """
    topic = rospy.get_param("~subscriber/" + name + "/topic")
    queue_size = rospy.get_param("~subscriber/" + name + "/queue_size")
    return topic, queue_size


def read_publisher_param(name):
    """
    @brief      Convenience function to read publisher parameter.
    """
    topic = rospy.get_param("~publisher/" + name + "/topic")
    queue_size = rospy.get_param("~publisher/" + name + "/queue_size")
    latch = rospy.get_param("~publisher/" + name + "/latch")
    return topic, queue_size, latch
