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

import json
import sys
import os
import glob

from logging import raiseExceptions
from re import A

import copy
import argparse
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import auc
import numpy as np
sys.path.append(".")


# laser to base
_ROT_Z = np.pi / 120
_R_laser_to_base = np.array([[np.cos(_ROT_Z), -np.sin(_ROT_Z), 0],
                            [np.sin(_ROT_Z), np.cos(_ROT_Z), 0],
                            [0, 0, 1]],
                            dtype=np.float32).T


def load_detection_file(kitti):
    locations = []
    classifications = []
    occluded = []

    if kitti:
        lines = kitti.split("\n")
        for line in lines:
            vals = line.split(" ")
            locations.append((float(vals[-5]), -float(vals[-4])))
            classifications.append(float(vals[-1]))
            occluded.append(int(vals[2]))

    locations = np.array(locations, dtype=np.float32)
    classifications = np.array(classifications, dtype=np.float32)
    occluded = np.array(occluded, dtype=np.int32)

    return locations, classifications, occluded


def get_target_cls_et_reg(
    scan_rphi, dets_rphi, person_rad_petit, person_rad_grand
):
    _, _num_scans = scan_rphi.shape

    # no annotation in this frame
    if len(dets_rphi) == 0:
        tar_cls = np.zeros(_num_scans, dtype=np.int64)
        tar_reg = np.zeros((_num_scans, 2), dtype=np.float32)
        anns_mask = []
    # this frame contains annotations
    else:
        min_adj_pts = 5
        scan_r, scan_phi = scan_rphi[0], scan_rphi[1]
        det_r, det_phi = dets_rphi[0], dets_rphi[1]
        scan_x = np.expand_dims(scan_r * np.cos(scan_phi), axis=0)
        scan_y = np.expand_dims(scan_r * np.sin(scan_phi), axis=0)
        dets_x = np.expand_dims(det_r * np.cos(det_phi), axis=1)
        dets_y = np.expand_dims(det_r * np.sin(det_phi), axis=1)

        pairwise_dist = np.hypot(scan_x - dets_x, scan_y - dets_y)

        # mark out annotations that has too few scan points
        anns_mask = (
            np.sum(pairwise_dist < person_rad_petit, axis=1) > min_adj_pts
        )  # (M, )

        # for each point, find the distance to its closest annotation
        argmin_pairwise_dist = np.argmin(pairwise_dist, axis=0)  # (n_scan, )
        min_pairwise_dist = pairwise_dist[argmin_pairwise_dist, np.arange(_num_scans)]

        # points within small radius, whose corresponding annotation is valid, is marked
        # as foreground
        tar_cls = -1 * np.ones(_num_scans, dtype=np.int64)
        valid_mask = np.logical_and(
            anns_mask[argmin_pairwise_dist], min_pairwise_dist < person_rad_petit
        )
        tar_cls[valid_mask] = 1
        tar_cls[min_pairwise_dist > person_rad_grand] = 0

        # regression target
        dets_matched_r, dets_matched_phi = dets_rphi[:, argmin_pairwise_dist]
        tar_reg_x = np.sin(dets_matched_phi - scan_phi) * dets_matched_r
        tar_reg_y = np.cos(dets_matched_phi - scan_phi) * dets_matched_r - scan_r
        tar_reg = np.stack([tar_reg_x, tar_reg_y], axis=1)

    return tar_cls, tar_reg, anns_mask


def get_precision_recall(
    dets_xy, dets_cls, dets_inds, gts_xy, gts_inds, ar
):
    a_rad = ar * np.ones(len(gts_inds), dtype=np.float32)

    frames = np.unique(np.r_[dets_inds, gts_inds])
    print("_prec_rec_2d:", dets_cls.shape, gts_inds.shape, len(frames))
    det_accepted_idxs = defaultdict(list)
    tps = np.zeros(len(frames), dtype=np.uint32)
    fps = np.zeros(len(frames), dtype=np.uint32)
    fns = np.array([np.sum(gts_inds == f) for f in frames], dtype=np.uint32)

    precisions = np.full_like(dets_cls, np.nan)
    recalls = np.full_like(dets_cls, np.nan)
    threshs = np.full_like(dets_cls, np.nan)

    indices = np.argsort(dets_cls, kind="mergesort")  
    for i, idx in enumerate(reversed(indices)):
        frame_tp = dets_inds[idx]
        iframe_tp = np.where(frames == frame_tp)[0][0]  

        # Accept this detection
        dets_idxs = det_accepted_idxs[frame_tp]
        dets_idxs.append(idx)
        threshs[i] = dets_cls[idx]

        dets = dets_xy[dets_idxs]
        print(dets)
        gts_mask = gts_inds == frame_tp
        gts = gts_xy[gts_mask]
        radii = a_rad[gts_mask]

        if len(gts) == 0:  
            fps[iframe_tp] += 1
        else:
            not_in_radius = radii[:, None] < cdist(gts, dets)
            igt, idet = linear_sum_assignment(not_in_radius)
            tps[iframe_tp] = np.sum(np.logical_not(not_in_radius[igt, idet]))
            fps[iframe_tp] = (len(dets) - tps[iframe_tp])
            fns[iframe_tp] = len(gts) - tps[iframe_tp]

        tp, fp, fn = np.sum(tps), np.sum(fps), np.sum(fns)
        precisions[i] = tp / (fp + tp) if fp + tp > 0 else np.nan
        recalls[i] = tp / (fn + tp) if fn + tp > 0 else np.nan

    print(precisions, recalls)
    
    assert np.sum(np.diff(recalls) >= 0) == len(precisions) - 1
    ap = auc(recalls, precisions)
    peak_f1 = np.max(2 * precisions * recalls / np.clip(precisions + recalls, 1e-16, 2 + 1e-16))
    eer = get_eer(recalls, precisions)
    return {
        "precisions": precisions,
        "recalls": recalls,
        "thresholds": threshs,
        "ap": ap,
        "peak_f1": peak_f1,
        "eer": eer,
    }


def stack_consecutive_lasers(data_path, seq_name, scan_url, _num_scans, scan_stride=1):

    scan_url = scan_url.replace("\\", "/")
    current_frame_idx = int(os.path.basename(scan_url).split(".")[0])
    frames_list = []
    for del_idx in range(_num_scans-1, 0-1, -1):
        frame_idx = max(0, current_frame_idx - del_idx * scan_stride)
        laser_url = os.path.join(os.path.dirname(scan_url), str(frame_idx).zfill(6) + ".txt").replace("\\", "/")
        laser_loaded = np.loadtxt(os.path.join(data_path, seq_name, laser_url), dtype=np.float32)
        frames_list.append(laser_loaded)
    return np.stack(frames_list, axis=0)


def get_eer(recs, precs):

    def first_nonzero_element(arr):
        return np.where(arr != 0)[0][0]

    p1 = first_nonzero_element(precs)
    r1 = first_nonzero_element(recs)
    idx = np.argmin(np.abs(precs[p1:] - recs[r1:]))
    return np.average([precs[p1 + idx], recs[r1 + idx]])


def launch_evaluate(_result_dir, _dataset_dir, _num_scans):

    det_dir = os.path.join(_result_dir, "detections")
    laser_dir = os.path.join(_dataset_dir, "lasers")

    seqs = os.listdir(det_dir)
    seq_03 = []
    seq_05 = []

    seqs_dets_xy = []
    seqs_dets_cls = []
    seqs_dets_inds = []
    seqs_gts_xy = []
    seqs_gts_inds = []

    # evaluate each sequence
    for seq_name in seqs:

        print(f"Evaluating {seq_name}")

        dets_xy_accum = []
        dets_cls_accum = []
        dets_inds_accum = []
        gts_xy_accum = []
        gts_inds_accum = []

        pc_file = os.path.join(_dataset_dir,
                                "labels",
                                "labels_3d",
                                f"{seq_name}.json")
        with open(pc_file, "r") as f:
            pc_labels = json.load(f)["labels"]

        timestamps_file = os.path.join(_dataset_dir,
                                        "timestamps",
                                        seq_name,
                                        "frames_pc_im_laser.json")
        with open(timestamps_file, "r") as f:
            frames_ts = json.load(f)["data"]

        # filter out 3D-annotated frames
        frames_labeled = []
        for frame in frames_ts:
            pc_file = os.path.basename(frame["pc_frame"]["pointclouds"][0]["url"])
            if pc_file in pc_labels:
                frames_labeled.append(frame)

        path_tmp = os.path.join(det_dir, seq_name, "*.txt")
        # accumulate detections in all frames
        for idx, (det_file, frame_seg) in enumerate(zip(glob.glob(path_tmp), frames_ts)):
            counter = idx + 1
            with open(det_file, "r") as f:
                dets_xy, dets_cls, _ = load_detection_file(f.read())

            # match labels by id
            file_id = os.path.basename(det_file)
            laser_url = frame_seg['laser_frame']['url'].split('\\')[-1]
            # load consecutive lasers
            laser_data = stack_consecutive_lasers(laser_dir, seq_name, laser_url, _num_scans)
            laser_grid = np.linspace(-np.pi, np.pi, max(laser_data.shape), dtype=np.float32)
            scan_r = np.stack((laser_data[-1][::-1], laser_grid), axis=0)

            pc_ref_id = os.path.basename(frames_labeled[idx]["pc_frame"]["pointclouds"][0]["url"])
            pc_anns = copy.deepcopy(pc_labels[pc_ref_id])
            print(idx, os.path.basename(det_file), pc_ref_id, laser_url)
            ann_coord = [
                (ann["box"]["cx"], ann["box"]["cy"], ann["box"]["cz"])
                for ann in pc_anns
            ] # ref to dataset: ann_xyz

            if len(ann_coord) > 0: # detection_of_walking_people
                ann_coord = np.array(ann_coord, dtype=np.float32).T
                ann_coord = _R_laser_to_base @ ann_coord
                ann_coord[1] = -ann_coord[1]
                ann_r, ann_phi = np.hypot(ann_coord[0], ann_coord[1]), np.arctan2(ann_coord[1], ann_coord[0])
                label_r = np.stack((ann_r, ann_phi), axis=0)
            else:
                label_r = []
            tar_cls, tar_reg, anns_valid_mask = get_target_cls_et_reg(scan_r, label_r, 0.4, 0.8)

            # to be consistant with DROWDataset
            dets_wp = [(label_r[0, i], label_r[1, i]) for i in range(label_r.shape[1])]

            if len(dets_wp) > 0:
                anns_rphi = np.array(dets_wp, dtype=np.float32)
                gts_x, gts_y = anns_rphi[:, 0] * np.cos(anns_rphi[:, 1]), anns_rphi[:, 0] * np.sin(anns_rphi[:, 1])
                gts_xy = np.stack((gts_x, gts_y), axis=1)
                gts_occluded = np.logical_not(anns_valid_mask).astype(np.int32)
            else:
                gts_xy = ""

            gts_xy = gts_xy[gts_occluded == 0]

            if len(dets_xy) > 0:
                dets_xy_accum.append(dets_xy)
                dets_cls_accum.append(dets_cls)
                dets_inds_accum += [counter] * len(dets_xy)

            if len(gts_xy) > 0:
                gts_xy_accum.append(gts_xy)
                gts_inds_accum += [counter] * len(gts_xy)

        dets_xy = np.concatenate(dets_xy_accum, axis=0)
        dets_cls = np.concatenate(dets_cls_accum)
        dets_inds = np.array(dets_inds_accum)
        gts_xy = np.concatenate(gts_xy_accum, axis=0)
        gts_inds = np.array(gts_inds_accum)

        print(dets_xy.shape, dets_inds.shape, gts_xy.shape, gts_inds.shape)

        # evaluate sequence
        seq_03.append(
            get_precision_recall(
                dets_xy, dets_cls, dets_inds, gts_xy, gts_inds, ar=0.3,
            )
        )

        seq_05.append(
            get_precision_recall(
                dets_xy, dets_cls, dets_inds, gts_xy, gts_inds, ar=0.5,
            )
        )

        print(
                f"AP_0.3 {seq_03[-1]['ap']:4f} "
                f"peak-F1_0.3 {seq_03[-1]['peak_f1']:4f} "
                f"EER_0.3 {seq_03[-1]['eer']:4f}\n"
                f"AP_0.5 {seq_05[-1]['ap']:4f} "
                f"peak-F1_0.5 {seq_05[-1]['peak_f1']:4f} "
                f"EER_0.5 {seq_05[-1]['eer']:4f}\n"
            )

        # store sequence detections and groundtruth for dataset evaluation
        seqs_dets_xy.append(dets_xy)
        seqs_dets_cls.append(dets_cls)
        seqs_dets_inds.append(dets_inds)
        seqs_gts_xy.append(gts_xy)
        seqs_gts_inds.append(gts_inds)

    # evaluate all dataset
    if len(seqs) > 1:
        print("Evaluating full dataset")

        dets_xy = np.concatenate(seqs_dets_xy, axis=0)
        dets_cls = np.concatenate(seqs_dets_cls)
        dets_inds = np.concatenate(seqs_dets_inds)
        gts_xy = np.concatenate(seqs_gts_xy, axis=0)
        gts_inds = np.concatenate(seqs_gts_inds)

        seq_03.append(
            get_precision_recall(
                dets_xy, dets_cls, dets_inds, gts_xy, gts_inds, ar=0.3,
            )
        )

        seq_05.append(
            get_precision_recall(
                dets_xy, dets_cls, dets_inds, gts_xy, gts_inds, ar=0.5,
            )
        )

        print(
                f"AP_0.3 {seq_03[-1]['ap']:5f} "
                f"peak-F1_0.3 {seq_03[-1]['peak_f1']:5f} "
                f"EER_0.3 {seq_03[-1]['eer']:5f}\n"
                f"AP_0.5 {seq_05[-1]['ap']:5f} "
                f"peak-F1_0.5 {seq_05[-1]['peak_f1']:5f} "
                f"EER_0.5 {seq_05[-1]['eer']:5f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--ros_dets_dir",
                        type=str,
                        required=True,
                        help="dects save path for external eval (ROS ONLY)")
    parser.add_argument("--dataset_dir",
                        type=str,
                        required=True,
                        help="disprete gt path of dataloader for external eval (ROS ONLY)")
    parser.add_argument("--model",
                        type=str,
                        default="DROW3",
                        help="model type")
    args = parser.parse_args()
    result_dir = args.ros_dets_dir
    dataset_dir = args.dataset_dir
    if args.model == "DROW3":
        NUM_SCANS = 1
    elif args.model == "DR-SPAAM":
        NUM_SCANS = 10
    else:
        raiseExceptions
    launch_evaluate(result_dir, dataset_dir, NUM_SCANS)