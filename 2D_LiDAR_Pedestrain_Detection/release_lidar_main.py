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

import sys
import os
import stat

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from StreamManagerApi import StreamManagerApi, MxDataInput, MxBufferInput, StringVector
from StreamManagerApi import InProtobufVector, MxProtobufIn
import MxpiDataType_pb2 as MxpiDataType

from srcs.data_loader import get_dataloader, drow_dataset, jrdb_dataset
import srcs.utils.precision_recall as pru
import srcs.utils.utils as u


FLAGS = os.O_WRONLY | os.O_CREAT
MODES = stat.S_IWUSR | stat.S_IRUSR


def throw_err(data_path, data_type, split, seq_name, data_index, pipe_path):
    """ check input errors
    """
    try:
        print("Input Check...")
        dataloader_len = {}
        if data_path:
            print(data_path, "1 +++---")
            if os.path.exists(data_path) is False:
                raise Exception("Invalid dataset path.")
            elif os.path.basename(data_path) not in ["DROWv2", "JRDB"]:
                raise Exception(
                    "Unsupport Dataset. Help Info: DROWv2 OR JRDB.")
        if seq_name:
            print(seq_name, "2 +++---")
            if os.path.basename(data_path) == "DROWv2":
                seq_path = os.path.join(
                    data_path, split.split("_")[0], seq_name)
            else:

                seq_path = os.path.join(
                    data_path, "{}_dataset".format(
                        split.split("_")[0]), "lasers", seq_name)
            if os.path.exists(seq_path) is False:
                raise Exception("Invalid sequence path: {}.".format(seq_path))
        if data_type:
            print(data_type, "3 +++---")
            if os.path.basename(data_path) == "DROWv2":
                [(root, dirs, files)] = os.walk(
                    os.path.join(data_path, split.split("_")[0]))
                cur_data_type = [file_name for file_name in files if os.path.splitext(
                    file_name)[-1] == '.csv']
                for idx, name in enumerate(cur_data_type):
                    seq_wc = []
                    basename = os.path.basename(name)
                    with open(os.path.join(root, name.replace("csv", "wc"))) as f:
                        for line in f:
                            seq, _ = line.split(",", 1)
                            seq_wc.append(int(seq))
                    dataloader_len[basename] = len(seq_wc)
            elif os.path.basename(data_path) == "JRDB":
                laser_path = os.path.join(
                    data_path, "{}_dataset".format(
                        split.split("_")[0]), "lasers")
                fuse_path = os.walk(laser_path)
                fuse_path = list(fuse_path)[1:]
                file_names = [fuse_path[i][-1] for i in range(len(fuse_path))]
                subpath_names = [
                    os.path.basename(
                        fuse_path[i][0]) for i in range(
                        len(fuse_path))]
                file_names_sqz = []
                for s in file_names:
                    file_names_sqz.extend(s)
                cur_data_type = [laser_name for laser_name in file_names_sqz if os.path.splitext(
                    laser_name)[-1] == '.txt']
                for idx, (file_name, subpath_name) in enumerate(
                        zip(file_names, subpath_names)):
                    dataloader_len[subpath_name] = len(file_name)
            if len(cur_data_type) == 0:
                raise Exception(
                    "Invalid DataType. Help Info: test set must contain REAL files.")
        if data_index:
            print(data_index, dataloader_len.get(seq_name, "abc"), "4 +++---")
            if data_index > dataloader_len.get(seq_name, "abc") - 1 or data_index < 0:
                raise Exception(
                    "Invalid frame id. Help Info: The length of dataloader is {}.".format(
                        dataloader_len.get(seq_name, "abc")))
        if pipe_path:
            print(pipe_path, "5 +++---")
            if os.path.exists(pipe_path) is False:
                raise Exception(
                    "Invalid .pipeline path. Help Info: please check your .pipeline path.")
            else:
                with open(pipe_path, 'rb') as f:
                    pipe_b = f.read()
                pipe_str = str(pipe_b, encoding='utf-8')
                pipe_dic = eval(pipe_str)
                if os.path.exists(
                        pipe_dic['detection0']['mxpi_tensorinfer0']['props']['modelPath']) is False:
                    raise Exception(
                        "Invalid .om path. Help Info: please modify .om path in .pipeline.")

    except Exception as e:
        print(repr(e))
        sys.exit("Program Exist. Welcome to the next use :D\n")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cross_entropy_with_logits(x, y):
    return -np.sum(x * np.log(y + 1e-7))


def rooted_mean_squared_error(x, y):
    mse = 0.5 * np.sum((y - x)**2)
    return np.sqrt(np.sum(mse)) / len(y)


def save_detections(det_coords, det_scores, occluded):

    if det_scores is None:
        det_scores = np.ones(len(det_coords), dtype=np.float32)

    if occluded is None:
        occluded = np.zeros(len(det_coords), dtype=np.int32)

    long_str = ""
    for cls, xy, occ in zip(det_scores, det_coords, occluded):
        long_str += f"Pedestrian 0 {occ} 0 0 0 0 0 0 0 0 0 {xy[0]} {xy[1]} 0 0 {cls}\n"
    long_str = long_str.strip("\n")

    return long_str


def rphi_xy_convertor(rdius, ang):
    return rdius * np.cos(ang), rdius * np.sin(ang)


def can_glob_convertor(rdius, ang, dx, dy):
    tmp_y = rdius + dy
    tmp_phi = np.arctan2(dx, tmp_y)
    dets_phi = tmp_phi + ang
    dets_r = tmp_y / np.cos(tmp_phi)
    return dets_r, dets_phi


def plot_one_frame_beta(
    batch_dict,
    frame_idx,
    pred_cls=None,
    pred_reg=None,
    xlim=(-7, 7),
    ylim=(-7, 7),
):
    fig_handle = plt.figure(figsize=(10, 10))
    ax_handle = fig_handle.add_subplot(111)

    ax_handle.set_xlim(xlim[0], xlim[1])
    ax_handle.set_ylim(ylim[0], ylim[1])
    ax_handle.set_xlabel("x [m]")
    ax_handle.set_ylabel("y [m]")
    ax_handle.set_aspect("equal")
    ax_handle.set_title(f"{frame_idx}")

    # scan and cls label
    scan_r = batch_dict["scans"][-1]
    scan_phi = batch_dict["scan_phi"]

    # plot scan
    scan_x, scan_y = rphi_xy_convertor(scan_r, scan_phi)
    ax_handle.scatter(scan_x, scan_y, s=1, c="black")
    pred_r, pred_phi = can_glob_convertor(
        scan_r, scan_phi, pred_reg[:, 0], pred_reg[:, 1]
    )
    pred_x, pred_y = rphi_xy_convertor(pred_r, pred_phi)
    ax_handle.scatter(pred_x, pred_y, s=2, c="red")

    return fig_handle, ax_handle


def plot_one_frame(
    batch_dict,
    frame_idx,
    pred_cls=None,
    pred_reg=None,
    dets_cls=None,
    dets_xy=None,
    xlim=(-7, 7),
    ylim=(-7, 7),
):

    fig_handle = plt.figure(figsize=(10, 10))
    ax_handle = fig_handle.add_subplot(111)

    ax_handle.set_xlim(xlim[0], xlim[1])
    ax_handle.set_ylim(ylim[0], ylim[1])
    ax_handle.set_xlabel("x [m]")
    ax_handle.set_ylabel("y [m]")
    ax_handle.set_aspect("equal")
    ax_handle.set_title(f"{frame_idx}")

    # scan and cls label
    scan_r = batch_dict["scans"][frame_idx][-1] if frame_idx is not None else batch_dict["scans"][-1]
    scan_phi = batch_dict["scan_phi"][frame_idx] if frame_idx is not None else batch_dict["scan_phi"]
    target_cls = batch_dict["target_cls"][frame_idx] if frame_idx is not None else batch_dict["target_cls"]

    # plot scan
    scan_x, scan_y = rphi_xy_convertor(scan_r, scan_phi)
    ax_handle.scatter(scan_x[target_cls < 0],
                      scan_y[target_cls < 0], s=1, c="orange")
    ax_handle.scatter(scan_x[target_cls == 0],
                      scan_y[target_cls == 0], s=1, c="black")
    ax_handle.scatter(scan_x[target_cls > 0],
                      scan_y[target_cls > 0], s=1, c="green")

    # annotation
    if frame_idx is not None:
        ann = batch_dict["dets_wp"][frame_idx]
        ann_valid_mask = batch_dict["anns_valid_mask"][frame_idx]
    else:
        ann = batch_dict["dets_wp"]
        ann_valid_mask = batch_dict["anns_valid_mask"]
    if len(ann) > 0:
        ann = np.array(ann)
        det_x, det_y = rphi_xy_convertor(ann[:, 0], ann[:, 1])
        for x, y, valid in zip(det_x, det_y, ann_valid_mask):
            c = "blue" if valid else "orange"
            c = plt.Circle((x, y), radius=0.4, color=c, fill=False)
            ax_handle.add_artist(c)

    # regression target
    target_reg = batch_dict["target_reg"][frame_idx] if frame_idx is not None else batch_dict["target_reg"]
    dets_r, dets_phi = can_glob_convertor(
        scan_r, scan_phi, target_reg[:, 0], target_reg[:, 1]
    )
    dets_r = dets_r[target_cls > 0]
    dets_phi = dets_phi[target_cls > 0]
    dets_x, dets_y = rphi_xy_convertor(dets_r, dets_phi)
    ax_handle.scatter(dets_x, dets_y, s=10, c="blue")

    # regression result
    if dets_xy is not None and dets_cls is not None:
        ax_handle.scatter(dets_xy[:, 0], dets_xy[:, 1],
                          marker="x", s=40, c="black")

    if pred_cls is not None and pred_reg is not None:
        pred_r, pred_phi = can_glob_convertor(
            scan_r, scan_phi, pred_reg[:, 0], pred_reg[:, 1]
        )
        pred_x, pred_y = rphi_xy_convertor(pred_r, pred_phi)
        ax_handle.scatter(pred_x, pred_y, s=2, c="red")

    return fig_handle, ax_handle


def nms_predicted_center(
    scan_grid, phi_grid, pred_cls, pred_reg, min_dist=0.5
):
    assert len(pred_cls.shape) == 1

    pred_r, pred_phi = can_glob_convertor(
        scan_grid, phi_grid, pred_reg[:, 0], pred_reg[:, 1]
    )
    pred_xs, pred_ys = rphi_xy_convertor(pred_r, pred_phi)

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


def main():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--data_path",
                        type=str,
                        required=True,
                        help="dataset directory.")
    parser.add_argument("--pipe_store",
                        type=str,
                        required=True,
                        help="collecting pipeline.")
    parser.add_argument("--split",
                        type=str,
                        required=True,
                        help="test, test_nano, or val.")
    parser.add_argument("--visu",
                        type=bool,
                        required=True,
                        help="visulizing the detection results.")
    parser.add_argument("--seq_name",
                        type=str,
                        default=None,
                        help="sequence name for quick test.")
    parser.add_argument("--frame_id",
                        type=int,
                        default=None,
                        help="frame index for quick test.")
    args = parser.parse_args()

    # input check (data_path, data_type, split, seq_name, data_index,
    # pipe_path)
    throw_err(args.data_path,
              True,
              args.split,
              args.seq_name,
              args.frame_id,
              args.pipe_store)

    dataset_name = os.path.basename(args.data_path)
    model_name = os.path.basename(args.pipe_store).split('.')[
        0].split('_')[0].upper()

    unseen_frame = False
    if args.frame_id:
        if "DROW" in args.data_path:
            test_dataset = drow_dataset.DROWDataset(
                split=args.split,
                data_dir=[args.data_path, args.seq_name],
                scan_type=model_name)
            test_loader = [test_dataset.__getitem__(int(args.frame_id))]
            try:
                if len(test_loader) == 0:
                    raise Exception(
                        "Sorry we cannot visit this frame at this time.")
            except Exception as e:
                print(repr(e))
                sys.exit("Program Exist. Welcome to the next use :D\n")
        if "JRDB" in args.data_path:
            test_dataset = jrdb_dataset.JRDBDataset(
                split=args.split,
                jrdb_split=args.seq_name,
                data_dir_cfg=args.data_path,
                scan_type=model_name)
            frame_url = str(args.frame_id).zfill(6) + ".txt"
            test_loader = list(
                filter(
                    lambda test_sample: os.path.basename(
                        test_sample["laser_frame"]["url"].replace(
                            "\\",
                            "/")) == frame_url,
                    test_dataset))
            print("len of test_loader", len(test_loader))
            if len(test_loader) == 0:
                unseen_frame = True
                num_scans = 1 if model_name == "DROW" else 10
                scan_url = os.path.join(args.data_path,
                                        args.split.split("_")[0] + "_dataset",
                                        "lasers",
                                        args.seq_name,
                                        frame_url)
                current_frame_idx = int(
                    os.path.basename(scan_url).split(".")[0])
                frames_list = []
                for del_idx in range(num_scans, 0, -1):
                    frame_idx = max(0, current_frame_idx - del_idx * 1)
                    laser_url = os.path.join(
                        os.path.dirname(scan_url),
                        str(frame_idx).zfill(6) +
                        ".txt").replace(
                        "\\",
                        "/")
                    laser_loaded = np.loadtxt(
                        os.path.join(
                            args.data_path,
                            laser_url),
                        dtype=np.float32)
                    frames_list.append(laser_loaded)

                batch_dict = {}
                laser_data = np.stack(frames_list, axis=0)

                laser_grid = np.linspace(-np.pi, np.pi,
                                         laser_data.shape[1], dtype=np.float32)
                ct_kwargs = {
                    "win_width": 1.0,
                    "win_depth": 0.5,
                    "num_ct_pts": 56,
                    "pad_val": 29.99,
                }
                batch_dict["scans"] = laser_data
                batch_dict["scan_phi"] = laser_grid
                batch_dict["input"] = u.trim_the_scans(
                    laser_data, laser_grid, stride=1, **ct_kwargs,)
                test_loader.append(batch_dict)

    elif args.seq_name:
        test_loader = get_dataloader(
            split=args.split,
            batch_size=1,
            num_workers=1,  # avoid TypeError: Caught TypeError in DataLoader worker process 0
            shuffle=False,
            dataset_pth=[args.data_path, args.seq_name],
            scan_type=model_name,
        )
    else:
        test_loader = get_dataloader(
            split=args.split,  # "val", "test", "test_nano"
            batch_size=1,
            num_workers=1,  # avoid TypeError: Caught TypeError in DataLoader worker process 0
            shuffle=False,
            dataset_pth=args.data_path,
            scan_type=model_name,
        )

    # The following belongs to the SDK Process
    stream_manager_api = StreamManagerApi()
    # init stream manager
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    else:
        print("-----------------创建流管理StreamManager并初始化-----------------")

    # create streams by pipeline config file
    with open(args.pipe_store, 'rb') as f:
        print("-----------------正在读取读取pipeline-----------------")
        pipeline_str = f.read()
        print("-----------------成功读取pipeline-----------------")

    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)

    # Print error message
    if ret != 0:
        print(
            "-----------------Failed to create Stream, ret=%s-----------------" %
            str(ret))
    else:
        print(
            "-----------------Create Stream Successfully, ret=%s-----------------" %
            str(ret))
    # Stream name

    stream_name = b'detection0'

    in_plugin_id = 0
    tb_dict_list = []
    dets_xy_accum = {}
    dets_cls_accum = {}
    dets_inds_accum = {}
    gts_xy_accum = {}
    gts_inds_accum = {}
    fig_dict = {}

    for count, batch_dict in enumerate(test_loader):
        if count >= 1e9:
            break
        tensor_package_list = MxpiDataType.MxpiTensorPackageList()
        tensor_package = tensor_package_list.tensorPackageVec.add()
        tsor = batch_dict.get("input", "abc").astype('<f4')
        if args.frame_id:
            tsor = np.expand_dims(tsor, axis=0)
        try:
            if tsor.shape[1] != 450 and tsor.shape[1] != 1091:
                raise AssertionError(
                    "InputTensor shape does not match model inputTensors.")
        except AssertionError as e:
            print(repr(e))
            sys.exit("Program Exist. Welcome to the next use :D\n")
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

        key = "appsrc{}".format(in_plugin_id).encode('utf-8')
        protobuf_vec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = key
        protobuf.type = b'MxTools.MxpiTensorPackageList'
        protobuf.protobuf = tensor_package_list.SerializeToString()
        protobuf_vec.push_back(protobuf)

        ret = stream_manager_api.SendProtobuf(
            stream_name, in_plugin_id, protobuf_vec)
        # in_plugin_id currently fixed to 0 indicating the input port number
        if ret != 0:
            print("Failed to send data to stream.")
            exit()

        key_vec = StringVector()
        key_vec.push_back(b'mxpi_tensorinfer0')
        infer_result = stream_manager_api.GetProtobuf(stream_name, 0, key_vec)

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

        print("Launching eval...")
        batch_size = len(batch_dict.get("scans", "abc"))
        if unseen_frame:
            print("Skip eval...")
            fig, ax = plot_one_frame_beta(
                batch_dict, args.frame_id, pred_cls[0], pred_reg[0])
            fig_name = f"figs_wo_anno/{args.seq_name}/{str(args.frame_id).zfill(6)}.png"
            fig_save_dir = os.getcwd()
            fig_file = os.path.join(fig_save_dir, fig_name)
            os.makedirs(os.path.dirname(fig_file), exist_ok=True)
            fig.savefig(fig_file)
            plt.close(fig)
            break
        for ib in range(batch_size):
            if args.frame_id:
                scan = batch_dict.get("scans", "abc")
                scan_phi = batch_dict.get("scan_phi", "abc")
                tar_cls = batch_dict.get("target_cls", "abc")
                tar_reg = batch_dict.get("target_reg", "abc")
                pred_cls_sigmoid = sigmoid(pred_cls.squeeze())
                det_xy, det_cls, _ = nms_predicted_center(
                    scan[-1], scan_phi, pred_cls_sigmoid, pred_reg.squeeze())
                frame_id = batch_dict.get("frame_id", "abc")
                frame_id = f"{frame_id:06d}"
                sequence = batch_dict.get("sequence", "abc")
                # save det_xy, det_cls for evaluation
                anns_rphi = batch_dict.get("dets_wp", "abc")
                if len(anns_rphi) > 0:
                    anns_rphi = np.array(anns_rphi, dtype=np.float32)
                    gts_xy = np.stack(rphi_xy_convertor(
                        anns_rphi[:, 0], anns_rphi[:, 1]), axis=1)
                    gts_occluded = np.logical_not(
                        batch_dict.get("anns_valid_mask", "abc")).astype(
                        np.int32)
                else:
                    gts_xy = ""
            else:
                scan = batch_dict.get("scans", "abc")[ib]
                scan_phi = batch_dict.get("scan_phi", "abc")[ib]
                tar_cls = batch_dict.get("target_cls", "abc")[ib]
                tar_reg = batch_dict.get("target_reg", "abc")[ib]
                pred_cls_sigmoid = sigmoid(pred_cls[ib].squeeze())
                det_xy, det_cls, _ = nms_predicted_center(
                    scan[-1], scan_phi, pred_cls_sigmoid, pred_reg[ib])  # by batch
                frame_id = batch_dict.get("frame_id", "abc")[ib]
                frame_id = f"{frame_id:06d}"
                sequence = batch_dict.get("sequence", "abc")[ib]
                # save det_xy, det_cls for evaluation
                anns_rphi = batch_dict.get("dets_wp", "abc")[ib]
                if len(anns_rphi) > 0:
                    anns_rphi = np.array(anns_rphi, dtype=np.float32)
                    gts_xy = np.stack(rphi_xy_convertor(
                        anns_rphi[:, 0], anns_rphi[:, 1]), axis=1)
                    gts_occluded = np.logical_not(
                        batch_dict.get("anns_valid_mask", "abc")[ib]).astype(
                        np.int32)
                else:
                    gts_xy = ""
            if "JRDB" in args.data_path:
                seq_nname = args.seq_name if args.seq_name else args.split
                det_str = save_detections(
                    det_xy, det_cls, None)
                det_fname = os.path.join(
                    os.getcwd(),
                    f"outputs_{dataset_name}_{seq_nname}_{model_name}/detections/{sequence}/{frame_id}.txt")
                os.makedirs(os.path.dirname(det_fname), exist_ok=True)
                with os.fdopen(os.open(det_fname, FLAGS, MODES), "w") as fdo:
                    fdo.write(det_str)
                gts_str = save_detections(
                    gts_xy, None, gts_occluded)
                gts_fname = os.path.join(
                    os.getcwd(),
                    f"outputs_{dataset_name}_{seq_nname}_{model_name}/groundtruth/{sequence}/{frame_id}.txt")
                os.makedirs(os.path.dirname(gts_fname), exist_ok=True)
                with os.fdopen(os.open(gts_fname, FLAGS, MODES), "w") as fgo:
                    fgo.write(gts_str)

            if len(det_xy) > 0:  # if not empty
                try:
                    if sequence not in list(dets_cls_accum.keys()):
                        dets_xy_accum[sequence] = []
                        dets_cls_accum[sequence] = []
                        dets_inds_accum[sequence] = []
                    dets_xy_accum[sequence].append(det_xy)
                    dets_cls_accum[sequence].append(det_cls)
                    dets_inds_accum[sequence] += [frame_id] * len(det_xy)
                except KeyError:
                    print("Dict KeyError!")

            if len(gts_xy) > 0:  # if not empty
                try:
                    if sequence not in list(gts_xy_accum.keys()):
                        gts_xy_accum[sequence], gts_inds_accum[sequence] = [], []
                    gts_xy_accum[sequence].append(gts_xy)
                    gts_inds_accum[sequence] += [frame_id] * len(gts_xy)
                except KeyError:
                    print("Dict KeyError!")

        # do the following in batch
        tar_shape = np.prod(batch_dict.get("target_cls", "abc").shape)

        tar_cls = tar_cls.reshape(tar_shape)
        pred_cls = pred_cls.reshape(tar_shape)

        # accumulate detections in all frames
        if args.visu:             # save fig
            fig_save_dir = os.getcwd()
            # fig_handle and axis_handle
            if args.frame_id:
                fig, ax = plot_one_frame(
                    batch_dict, None, pred_cls[0], pred_reg[0], det_cls, det_xy)
                if args.seq_name:
                    fig_name = f"figs/{args.seq_name}/{frame_id}.png"
                else:
                    fig_name = f"figs/{sequence}/{frame_id}.png"
            else:
                fig, ax = plot_one_frame(
                    batch_dict, ib, pred_cls[ib], pred_reg[ib], det_cls, det_xy)
                fig_name = f"figs/{sequence}/{frame_id}.png"
            fig_file = os.path.join(
                fig_save_dir, fig_name)
            os.makedirs(os.path.dirname(fig_file), exist_ok=True)
            fig.savefig(fig_file)
            plt.close(fig)

    # evaluate
    if "DROW" in args.data_path and args.frame_id is None:
        for key in [*dets_cls_accum]:
            dets_xy = np.concatenate(dets_xy_accum.get(key, "abc"), axis=0)
            dets_cls = np.concatenate(dets_cls_accum.get(key, "abc"))
            dets_inds = np.array(dets_inds_accum.get(key, "abc"))
            gts_xy = np.concatenate(gts_xy_accum.get(key, "abc"), axis=0)
            gts_inds = np.array(gts_inds_accum.get(key, "abc"))
            # evaluate all sequences

            seq03, seq05 = [], []
            try:
                seq03.append(
                    pru.eval_internal(
                        dets_xy,
                        dets_cls,
                        dets_inds,
                        gts_xy,
                        gts_inds,
                        ar=0.3))
            except Exception as e:
                print(repr(e))
            try:
                seq05.append(
                    pru.eval_internal(
                        dets_xy,
                        dets_cls,
                        dets_inds,
                        gts_xy,
                        gts_inds,
                        ar=0.5))
            except Exception as e:
                print(repr(e))
            if len(seq03) > 0 and len(seq05) > 0:
                print(f"Evaluating {key} {args.frame_id}")
                print(
                    f"AP_0.3 {seq03[-1]['ap']:4f} "
                    f"peak-F1_0.3 {seq03[-1]['peak_f1']:4f} "
                    f"EER_0.3 {seq03[-1]['eer']:4f}\n"
                    f"AP_0.5 {seq05[-1]['ap']:4f} "
                    f"peak-F1_0.5 {seq05[-1]['peak_f1']:4f} "
                    f"EER_0.5 {seq05[-1]['eer']:4f}\n"
                )
            elif len(seq03) > 0:
                print(f"Evaluating ar=0.3 {key} {args.frame_id}")
                print(
                    f"AP_0.3 {seq03[-1]['ap']:4f} "
                    f"peak-F1_0.3 {seq03[-1]['peak_f1']:4f} "
                    f"EER_0.3 {seq03[-1]['eer']:4f}\n"
                )
            elif len(seq05) > 0:
                print(f"Evaluating ar=0.5 {key} {args.frame_id}")
                print(
                    f"AP_0.3 {seq05[-1]['ap']:4f} "
                    f"peak-F1_0.3 {seq05[-1]['peak_f1']:4f} "
                    f"EER_0.3 {seq05[-1]['eer']:4f}\n"
                )


if __name__ == "__main__":
    main()
