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
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset

from srcs.data_handle.jrdb_handle import JRDBv1Handle
import srcs.utils.utils as u


__all__ = ["JRDBDataset"]


# laser to base
_ROT_Z = np.pi / 120
_R_laser_to_base = np.array([[np.cos(_ROT_Z), -np.sin(_ROT_Z), 0], 
                            [np.sin(_ROT_Z), np.cos(_ROT_Z), 0], 
                            [0, 0, 1]], 
                            dtype=np.float32)


class JRDBDataset(Dataset):
    def __init__(self, split, jrdb_split, data_dir_cfg, scan_type):
        self._data_handle = { "data_dir": data_dir_cfg[0] if isinstance(data_dir_cfg, list) else data_dir_cfg,
                            "num_scans":  1 if scan_type == "DROW3" else 10,
                            "scan_stride": 1,
                            "tracking": False
        }

        self._jrdb_split = jrdb_split

        if split == "train":
            self.__handle = JRDBv1Handle(
                "train", self._data_handle, sequences=defaultdict(lambda: self._jrdb_split.get("train", "abc"))[0]
            )
        elif split == "val" or split == "test":
            self.__handle = JRDBv1Handle(
                "train", self._data_handle, sequences=defaultdict(lambda: self._jrdb_split.get("test", "abc"))[0]
            )
        elif split == "train_val":
            self.__handle = JRDBv1Handle("train", self._data_handle)
        elif split == "test_nano" or split == "train_nano":
            self.__handle = JRDBv1Handle(split.split("_")[0], self._data_handle, sequences=[data_dir_cfg[1]])
        elif split == "test_single" or split == "train_single":
            self.__handle = JRDBv1Handle(split.split("_")[0], self._data_handle, sequences=[self._jrdb_split])
        else:
            raise RuntimeError(f"Invalid split: {split}")

        self.__split = split

        self._augment_data = False 
        self._person_only = True 
        self._cutout_kwargs = {
                            "win_width": 1.0,
                            "win_depth": 0.5,
                            "num_ct_pts": 56,
                            "pad_val": 29.99,
                            } 
        self._pseudo_label = False 
        self._pl_correction_level = 0

    def __len__(self):
        return len(self.__handle)

    def __getitem__(self, idx):
        return self._next_jrdb_sample(idx)

    @property
    def split(self):
        return self.__split 

    @property
    def sequence_beginning_inds(self):
        return self.__handle.sequence_beginning_inds

    def collect_batch(self, batch):
        jrdb_rtn = {}
        for k, _ in batch[0].items():
            if k in [
                "target_cls",
                "target_reg",
                "input",
            ]:
                jrdb_rtn[k] = np.array([sample[k] for sample in batch])
            else:
                jrdb_rtn[k] = [sample[k] for sample in batch]

        return jrdb_rtn   

    def _next_jrdb_sample(self, idx):
        jrdb_set = self.__handle[idx]

        # DROW defines laser frame as x-forward, y-right, z-downward
        # JRDB defines laser frame as x-forward, y-left, z-upward
        # Use DROW frame for DR-SPAAM or DROW3

        # equivalent of flipping y axis (inversing laser phi angle)
        jrdb_set["laser_data"] = jrdb_set["laser_data"][:, ::-1]
        scan_rphi = np.stack(
            (jrdb_set["laser_data"][-1], jrdb_set["laser_grid"]), axis=0
        )

        # get annotation in laser frame
        pc_xyz = [
            (pc["box"]["cx"], pc["box"]["cy"], pc["box"]["cz"])
            for pc in jrdb_set["pc_anns"]
        ]
        if len(pc_xyz) > 0:
            pc_xyz = np.array(pc_xyz, dtype=np.float32).T
            pc_xyz = _R_laser_to_base.T @ pc_xyz
            pc_xyz[1] = -pc_xyz[1]  # to DROW frame
            dets_rphi = np.stack(u.xy_to_rphi(pc_xyz[0], pc_xyz[1]), axis=0)
        else:
            dets_rphi = []

        # regression target
        tar_cls, tar_reg, anns_valid_mask = _get_target_cls_et_reg(
            scan_rphi,
            dets_rphi,
            person_rad_petit=0.4,
            person_rad_grand=0.8,
        )

        jrdb_set["target_cls"] = tar_cls
        jrdb_set["target_reg"] = tar_reg
        jrdb_set["anns_valid_mask"] = anns_valid_mask


        # to be consistent with DROWDataset in order to use the same evaluation function
        dets_wp = []
        for i in range(dets_rphi.shape[1]):
            dets_wp.append((dets_rphi[0, i], dets_rphi[1, i]))
        jrdb_set["dets_wp"] = dets_wp
        jrdb_set["scans"] = jrdb_set["laser_data"]
        jrdb_set["scan_phi"] = jrdb_set["laser_grid"]


        jrdb_set["input"] = u.trim_the_scans(
            jrdb_set["laser_data"],
            jrdb_set["laser_grid"],
            stride=1,
            **self._cutout_kwargs,
        )

        return jrdb_set



def transform_pts_base_to_stitched_im(pts):
    
    im_size = (480, 3760)

    # to image coordinate
    pts_rect = pts[[1, 2, 0], :]
    pts_rect[:2, :] *= -1

    # to pixel
    horizontal_theta = np.arctan2(pts_rect[0], pts_rect[2])
    horizontal_percent = horizontal_theta / (2 * np.pi) + 0.5
    x = im_size[1] * horizontal_percent
    y = (
        485.78 * pts_rect[1] / pts_rect[2] * np.cos(horizontal_theta)
        + 0.4375 * im_size[0]
    )

    inbound_mask = y < im_size[0]

    return np.stack((x, y), axis=0).astype(np.int32), inbound_mask


def _get_target_cls_et_reg(
    scan_rphi, dets_rphi, person_rad_petit, person_rad_grand
):
    _, num_scans = scan_rphi.shape

    # no annotation in this frame
    if len(dets_rphi) == 0:
        tar_cls = np.zeros(num_scans, dtype=np.int64)
        tar_reg = np.zeros((num_scans, 2), dtype=np.float32)
        anns_mask = []
    
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
        min_pairwise_dist = pairwise_dist[argmin_pairwise_dist, np.arange(num_scans)]

        # points within small radius, whose corresponding annotation is valid, is marked
        # as foreground
        tar_cls = -1 * np.ones(num_scans, dtype=np.int64)
        valid_mask = np.logical_and(
            anns_mask[argmin_pairwise_dist], min_pairwise_dist < person_rad_petit
        )
        tar_cls[valid_mask] = 1
        tar_cls[min_pairwise_dist > person_rad_grand] = 0

        # regression target
        dets_matched_r = dets_rphi[:, argmin_pairwise_dist][0]
        dets_matched_phi = dets_rphi[:, argmin_pairwise_dist][1]
        tar_reg_x = np.sin(dets_matched_phi - scan_phi) * dets_matched_r
        tar_reg_y = np.cos(dets_matched_phi - scan_phi) * dets_matched_r - scan_r
        tar_reg = np.stack([tar_reg_x, tar_reg_y], axis=1)

    return tar_cls, tar_reg, anns_mask
