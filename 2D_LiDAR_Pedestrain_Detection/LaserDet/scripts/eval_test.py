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

import argparse
import numpy as np
from re import A

sys.path.append(".")
from collections import defaultdict
import glob


from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import auc


def load_detection_file(kitti):
    locations = []
    classifications = []
    occluded = []

    if kitti:
        lines = kitti.split("\n")
        for line in lines:
            vals = line.split(" ")
            locations.append((float(vals[-5]), float(vals[-4])))
            classifications.append(float(vals[-1]))
            occluded.append(int(vals[2]))

    locations = np.array(locations, dtype=np.float32)
    classifications = np.array(classifications, dtype=np.float32)
    occluded = np.array(occluded, dtype=np.int32)

    return locations, classifications, occluded


def get_precision_recall(
    dets_xy, dets_cls, dets_inds, gts_xy, gts_inds, ar
):
    a_rad = ar * np.ones(len(gts_inds), dtype=np.float32)

    frames = np.unique(np.r_[dets_inds, gts_inds])

    det_accepted_idxs = defaultdict(list)
    tps = np.zeros(len(frames), dtype=np.uint32)
    fps = np.zeros(len(frames), dtype=np.uint32)
    fns = np.array([np.sum(gts_inds == f) for f in frames], dtype=np.uint32)

    precisions = np.full_like(dets_cls, np.nan)
    recalls = np.full_like(dets_cls, np.nan)
    threshs = np.full_like(dets_cls, np.nan)

    indices = np.argsort(dets_cls, kind="mergesort")  # mergesort for determinism.
    for i, idx in enumerate(reversed(indices)):
        frame = dets_inds[idx]
        iframe = np.where(frames == frame)[0][0]  # Can only be a single one.

        # Accept this detection
        dets_idxs = det_accepted_idxs[frame]
        dets_idxs.append(idx)
        threshs[i] = dets_cls[idx]

        dets = dets_xy[dets_idxs]

        gts_mask = gts_inds == frame
        gts = gts_xy[gts_mask]
        radii = a_rad[gts_mask]

        if len(gts) == 0:  # No GT, but there is a detection.
            fps[iframe] += 1
        else:
            not_in_radius = radii[:, None] < cdist(gts, dets)
            igt, idet = linear_sum_assignment(not_in_radius)

            tps[iframe] = np.sum(np.logical_not(not_in_radius[igt, idet]))
            fps[iframe] = (len(dets) - tps[iframe])
            fns[iframe] = len(gts) - tps[iframe]

        tp, fp, fn = np.sum(tps), np.sum(fps), np.sum(fns)
        precisions[i] = tp / (fp + tp) if fp + tp > 0 else np.nan
        recalls[i] = tp / (fn + tp) if fn + tp > 0 else np.nan


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


def get_eer(recs, precs):

    def first_nonzero_element(arr):
        return np.where(arr != 0)[0][0]

    p1 = first_nonzero_element(precs)
    r1 = first_nonzero_element(recs)
    idx = np.argmin(np.abs(precs[p1:] - recs[r1:]))
    return np.average([precs[p1 + idx], recs[r1 + idx]])


def launch_evaluate(_result_dir):

    det_dir = os.path.join(_result_dir, "detections")

    seqs = os.listdir(det_dir)
    seq_03 = []
    seq_05 = []

    seqs_dets_xy = []
    seqs_dets_cls = []
    seqs_dets_inds = []
    seqs_gts_xy = []
    seqs_gts_inds = []

    counter = 0
    # evaluate each sequence
    for seq_name in seqs:

        print(f"Evaluating {seq_name}")

        dets_xy_accum = []
        dets_cls_accum = []
        dets_inds_accum = []
        gts_xy_accum = []
        gts_inds_accum = []

        # accumulate detections in all frames
        for det_file in glob.glob(os.path.join(det_dir, seq_name, "*.txt")):
            counter += 1

            with open(det_file, "r") as f:
                dets_xy, dets_cls, _ = load_detection_file(f.read())

            with open(det_file.replace("detections", "groundtruth"), "r") as f:
                gts_xy, _, gts_occluded = load_detection_file(f.read())

            # evaluate only on visiable groundtruth
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
    parser.add_argument("--result_dir",
                        type=str,
                        required=True,
                        help="dects and gts save path for external eval lol lmao XD")
    args = parser.parse_args()
    result_dir = args.result_dir
    launch_evaluate(result_dir)
