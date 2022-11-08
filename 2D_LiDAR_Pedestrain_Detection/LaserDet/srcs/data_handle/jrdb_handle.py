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
import copy
import os
import json
import cv2
import numpy as np


# Force the dataloader to load only one sample, in which case the network should
# fit perfectly.
# Pointcloud and image is only needed for visualization. Turn off for fast dataloading

__all__ = ["JRDBv1Handle"]


class JRDBv1Handle:
    def __init__(self, split, cfg, sequences=None, exclude_sequences=None):

        self.__num_scans = cfg["num_scans"]
        self.__scan_stride = cfg["scan_stride"]

        data_path = os.path.abspath(os.path.expanduser(cfg["data_dir"]))
        data_path = (
            os.path.join(data_path, "train_dataset")
            if split == "train" or split == "val"
            else os.path.join(data_path, "test_dataset")
        )
        print("JRDB data_path", cfg["data_dir"], data_path)
        self.data_path = data_path
        self.timestamp_path = os.path.join(data_path, "timestamps")
        self.pc_label_path = os.path.join(data_path, "labels", "labels_3d")
        self.im_label_path = os.path.join(data_path, "labels", "labels_2d_stitched")

        if sequences is not None:
            seq_names = sequences
        else:
            seq_names = os.listdir(self.timestamp_path)
            # NOTE it is important to sort the return of os.listdir, since its order
            # changes for different file system.
            seq_names.sort()

        if exclude_sequences is not None:
            seq_names = [s for s in seq_names if s not in exclude_sequences]

        self.seq_names = seq_names

        self.seq_handle = []
        self._seq_init_start = [0]
        self.__seq_inds_inline = []
        self.__frame_inds_inline = []
        for seq_idx, seq_name in enumerate(self.seq_names):
            self.seq_handle.append(_SequenceHandle(self.data_path, seq_name))

            # build a flat index for all sequences and frames
            seq_len = len(self.seq_handle[-1])
            self.__seq_inds_inline += seq_len * [seq_idx]
            self.__frame_inds_inline += range(seq_len)

            self._seq_init_start.append(
                self._seq_init_start[-1] + seq_len
            )

    def __len__(self):
        return len(self.__frame_inds_inline)

    def __getitem__(self, idx):
        idx_sq = self.__seq_inds_inline[idx]
        idx_fr = self.__frame_inds_inline[idx]
        handle_ind = self.seq_handle[idx_sq][idx_fr]
        get_frame_dict = handle_ind.frame

        pc_data = {}
        im_data = {}
        for pc_dict in get_frame_dict["pc_frame"]["pointclouds"]: # 3D annotation is not supported at this moment
            pc_data[pc_dict["name"]] = 0

        for im_dict in get_frame_dict["im_frame"]["cameras"]:
            im_data[im_dict["name"]] = cv2.cvtColor(cv2.imread(os.path.join(self.data_path,
                                                                            im_dict["url"].replace("\\", "/")),
                                                                cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)

        laser_data = self._load_consecutive_lasers(get_frame_dict["laser_frame"]["url"].replace("\\", "/"))
        laser_grid = np.linspace(-np.pi, np.pi, laser_data.shape[1], dtype=np.float32)
        laser_z = -0.5 * np.ones(laser_data.shape[1], dtype=np.float32)
        get_frame_dict.update(
            {
                "frame_id": int(get_frame_dict["frame_id"]),
                "sequence": self.seq_handle[idx_sq].sequence,
                "first_frame": idx_fr == 0,
                "idx": idx,
                "pc_data": pc_data,
                "im_data": im_data,
                "laser_data": laser_data,
                "pc_anns": handle_ind.pc_anns,
                "im_anns": handle_ind.img_anns,
                "im_dets": handle_ind.img_dets,
                "laser_grid": laser_grid,
                "laser_z": laser_z,
            }
        )
        return get_frame_dict

    @property
    def seq_init_start(self):
        return copy.deepcopy(self._seq_init_start)


    def _load_consecutive_lasers(self, scan_url):
        """Load current and previous consecutive laser scans.

        Args:
            url (str): file url of the current scan

        Returns:
            pc (np.ndarray[self.num_scan, N]): Forward in time with increasing
                row index, i.e. the latest scan is pc[-1]
        """
        scan_url = scan_url.replace("\\", "/")
        current_frame_idx = int(os.path.basename(scan_url).split(".")[0])
        frames_list = []
        for del_idx in range(self.__num_scans, 0, -1):
            frame_idx = max(0, current_frame_idx - del_idx * self.__scan_stride)
            laser_url = os.path.join(os.path.dirname(scan_url), str(frame_idx).zfill(6) + ".txt").replace("\\", "/")
            laser_loaded = np.loadtxt(os.path.join(self.data_path, laser_url), dtype=np.float32)
            frames_list.append(laser_loaded)

        return np.stack(frames_list, axis=0)


class _FrameHandle:
    def __init__(self, frame, pc_anns, img_anns, img_dets):
        self.frame = frame
        self.pc_anns = pc_anns
        self.img_anns = img_anns
        self.img_dets = img_dets


class _SequenceHandle:
    def __init__(self, data_path, sequence):
        self.sequence = sequence

        # load frames of the sequence
        fname = os.path.join(data_path,
                            "timestamps",
                            self.sequence,
                            "frames_pc_im_laser.json")
        with open(fname, "r") as f:
            self.frames = json.load(f)["data"]

        # load 3D annotation pointcloud
        pc_fname = os.path.join(data_path,
                            "labels",
                            "labels_3d",
                            f"{self.sequence}.json")
        with open(pc_fname, "r") as f:
            self.pc_labels = json.load(f)["labels"]

        # load 2D annotation
        im_label_fname = os.path.join(data_path,
                            "labels",
                            "labels_2d_stitched",
                            f"{self.sequence}.json")
        with open(im_label_fname, "r") as f:
            self.im_labels = json.load(f)["labels"]

        # load 2D detection
        im_det_fname = os.path.join(data_path,
                            "detections",
                            "detections_2d_stitched",
                            f"{self.sequence}.json")
        with open(im_det_fname, "r") as f:
            self.im_dets = json.load(f)["detections"]

        # find out which frames has 3D annotation
        self.frames_labeled = []
        for frame in self.frames:
            pc_file = os.path.basename(frame["pc_frame"]["pointclouds"][0]["url"].replace("\\", "/"))

            if pc_file in self.pc_labels:
                self.frames_labeled.append(frame)

        self.jrdb_frames = (
            self.frames_labeled
        )


    def __len__(self):
        return len(self.jrdb_frames)

    def __getitem__(self, idx):
        # NOTE It's important to use a copy as the return dict, otherwise the
        # original dict in the data handle will be corrupted
        jrdb_frame = copy.deepcopy(self.jrdb_frames[idx])

        if self._is_unlabeled_frames:
            frame_handle = _FrameHandle(jrdb_frame, [], [], [])
            return frame_handle     # frame, [], [], []

        pc_load = os.path.basename(jrdb_frame["pc_frame"]["pointclouds"][0]["url"].replace("\\", "/"))
        pc_anns = copy.deepcopy(self.pc_labels[pc_load])

        img_file = os.path.basename(jrdb_frame["im_frame"]["cameras"][0]["url"].replace("\\", "/"))
        img_anns = copy.deepcopy(self.im_labels[img_file])
        img_dets = copy.deepcopy(self.im_dets[img_file])

        frame_handle = _FrameHandle(jrdb_frame, pc_anns, img_anns, img_dets)

        return frame_handle
