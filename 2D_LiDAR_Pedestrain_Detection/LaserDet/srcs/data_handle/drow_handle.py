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
from glob import glob
import os
import json
import numpy as np


__all__ = ["DROWv2Handle"]


class DROWv2Handle:
    def __init__(self, split, data_cfg):
        assert split in ["train", "val", "test",
                        "test_nano", "test_single",
                        "train_nano", "train_single"], f'Invalid split "{split}"'

        self.__n_scans = data_cfg["num_scans"]
        self.__scan_stride = data_cfg["scan_stride"]

        if len(data_cfg["data_dir"]) > 1 and isinstance(data_cfg["data_dir"], str) is not True:
            drow_path = os.path.abspath(os.path.expanduser(data_cfg["data_dir"][0]))
            self.seq_names = [os.path.join(drow_path, split.split("_")[0], data_cfg["data_dir"][1])[:-4]]
            print("***", data_cfg["data_dir"], drow_path, self.seq_names)
        else:
            drow_path = os.path.abspath(os.path.expanduser(data_cfg["data_dir"]))
            self.seq_names = [f[:-4] for f in glob(os.path.join(drow_path, split, "*.csv"))]

        # preload all annotations
        self.dets_seq_wc, self.wcs, self.was, self.wps = list(zip(
            *map(lambda f: self._preload(f), self.seq_names)
        ))

        # look-up list to convert an index to sequence index and annotation index
        self.__seq_inds_inline = []
        self.__det_inds_inline = []
        for seq_index, det_ns in enumerate(self.dets_seq_wc):
            self.__seq_inds_inline += [seq_index] * len(det_ns)
            self.__det_inds_inline += range(len(det_ns))

        # placeholder for scans, which will be preload on the fly
        self.seq_len = len(self.seq_names)
        self.scans_seq_no = [None] * self.seq_len
        self.scans_seq_t = [None] * self.seq_len
        self.scans_seq_dist = [None] * self.seq_len

        # placeholder for mapping from detection index to scan index
        self.__did2sid = [None] * self.seq_len

        # load the scan sequence into memory if it has not been loaded
        for seq_index, s in enumerate(self.seq_names):
            if self.scans_seq_dist[seq_index] is None:
                self._get_scan_seq(seq_index)


    def __len__(self):
        return len(self.__det_inds_inline)


    def __getitem__(self, idx):
        # find matching seq_index, det_index, and scan_index
        seq_index = self.__seq_inds_inline[idx]
        det_index = self.__det_inds_inline[idx]
        scan_index = self.__did2sid[seq_index][det_index]

        drow_rtn = {
            "idx": idx,
            "dets_wc": self.wcs[seq_index][det_index],
            "dets_wa": self.was[seq_index][det_index],
            "dets_wp": self.wps[seq_index][det_index],
        }

        # load sequential scans up to the current one (array[frame, point])
        delta_inds = (np.arange(self.__n_scans) * self.__scan_stride)[::-1]
        scans_inds = [max(0, scan_index - i) for i in delta_inds]
        drow_rtn["scans"] = np.array([self.scans_seq_dist[seq_index][i] for i in scans_inds])
        drow_rtn["scans_ind"] = scans_inds
        laser_fov = (450 - 1) * np.radians(0.5)
        drow_rtn["scan_phi"] = np.linspace(-laser_fov * 0.5, laser_fov * 0.5, 450)

        return drow_rtn


    @classmethod
    def _preload(self, seq_name):
        def get_f(f_name):
            seqs, dets = [], []
            with open(f_name) as f:
                for line in f:
                    seq, tail = line.split(",", 1)
                    seqs.append(int(seq))
                    dets.append(json.loads(tail))
            return seqs, dets

        seq_wc, wcs = get_f(seq_name + ".wc") # [[]*N]
        seq_wa, was = get_f(seq_name + ".wa") # [[]*N]
        seq_wp, wps = get_f(seq_name + ".wp") # if not empty
        assert all(wc == wa == wp for wc, wa, wp in zip(seq_wc, seq_wa, seq_wp))

        return [np.array(seq_wc), wcs, was, wps]


    def _get_scan_seq(self, seq_index):
        drow_data = np.genfromtxt(self.seq_names[seq_index] + ".csv", delimiter=",")
        self.scans_seq_no[seq_index] = drow_data[:, 0].astype(np.uint32)
        self.scans_seq_t[seq_index] = drow_data[:, 1].astype(np.float32)
        self.scans_seq_dist[seq_index] = drow_data[:, 2:].astype(np.float32)

        is_ = 0
        id2is = []
        for det_ns in self.dets_seq_wc[seq_index]:
            while self.scans_seq_no[seq_index][is_] != det_ns:
                is_ += 1
            id2is.append(is_)
        self.__did2sid[seq_index] = id2is
