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
import numpy as np
from torch.utils.data import Dataset

from srcs.data_handle.drow_handle import DROWv2Handle
import srcs.utils.utils as u


class DROWDataset(Dataset):
    def __init__(self, split, data_dir, scan_type):
        self._data_handle = {"data_dir": data_dir,
                            "num_scans": 1 if scan_type == "DROW3" else 10,
                            "scan_stride": 1}
        self._cutout_kwargs = {
                            "win_width": 1.0,
                            "win_depth": 0.5,
                            "num_ct_pts": 56,
                            "pad_val": 29.99,
                            }
        self.__handle = DROWv2Handle(split, self._data_handle)
        self.__split = split

        self._augment_data = False 
        self._pedestrain_only = True 
        self._cutout_kwargs = self._cutout_kwargs

    def __getitem__(self, idx):
        drow_set = self.__handle[idx]

        # regression target
        tar_cls, tar_reg = _get_target_cls_et_reg(
            drow_set["scans"][-1],
            drow_set["scan_phi"],
            drow_set["dets_wc"],
            drow_set["dets_wa"],
            drow_set["dets_wp"],
            0.6,
            0.4,
            0.35,
            1,
            2,
            3,
            self._pedestrain_only,
        )

        drow_set["target_cls"] = tar_cls
        drow_set["target_reg"] = tar_reg


        drow_set["input"] = u.trim_the_scans(
            drow_set["scans"], drow_set["scan_phi"], stride=1, **self._cutout_kwargs
        )

        # to be consistent with JRDB dataset
        drow_set["frame_id"] = drow_set["idx"]
        drow_set["sequence"] = "all"

        # this is used by JRDB dataset to mask out annotations, to be consistent
        drow_set["anns_valid_mask"] = np.ones(len(drow_set["dets_wp"]), dtype=np.bool)

        return drow_set
    
    def __len__(self):
        return len(self.__handle)
        
    @property
    def split(self):
        return self.__split  # used by trainer.py

    def collect_batch(self, batch):
        drow_rtn = {}
        for k, _ in batch[0].items():
            if k in ["target_cls", "target_reg", "input"]:
                drow_rtn[k] = np.array([sample[k] for sample in batch])
            else:
                drow_rtn[k] = [sample[k] for sample in batch]

        return drow_rtn


def _get_target_cls_et_reg(
    scan,
    scan_phi,
    wcs,
    was,
    wps,
    radii_wc,
    radii_wa,
    radii_wp,
    gts_wc,
    gts_wa,
    gts_wp,
    pedestrain_only,
):
    tol_points = len(scan)
    tar_cls = np.zeros(tol_points, dtype=np.int64)
    tar_reg = np.zeros((tol_points, 2), dtype=np.float32)

    if pedestrain_only:
        all_dets = list(wps)
        all_radius = [radii_wp] * len(wps)
        labels = [0] + [1] * len(wps)
    else:
        all_dets = list(wcs) + list(was) + list(wps)
        all_radius = (
            [radii_wc] * len(wcs) + [radii_wa] * len(was) + [radii_wp] * len(wps)
        )
        labels = (
            [0] + [gts_wc] * len(wcs) + [gts_wa] * len(was) + [gts_wp] * len(wps)
        )

    dets_refined = _nearest_result(scan, scan_phi, all_dets, all_radius)
    # reshape list
    dets_refined = np.asarray(dets_refined).squeeze()

    for i, (r, phi) in enumerate(zip(scan, scan_phi)):
        if 0 < dets_refined[i]:
            tar_cls[i] = labels[dets_refined[i]]
            tar_reg[i, :] = u.global_to_canonical(r, phi, *all_dets[dets_refined[i] - 1])

    return tar_cls, tar_reg


def _nearest_result(scan, scan_phi, dets, radii):
    
    if len(dets) == 0:
        return np.zeros_like(scan, dtype=int)

    assert len(dets) == len(radii), "Need to give a radius for each detection!"

    # Distance (in x,y space) of each laser-point with each detection.
    scans_xy = np.array(u.rphi_to_xy(scan, scan_phi)).T  # (N, 2)
    dets_xy = np.array([u.rphi_to_xy(r, phi) for r, phi in dets]) # (N,2) same here
    mat_mul = np.dot(scans_xy, dets_xy.T)
    te = np.square(scans_xy).sum(axis=1)
    tr = np.square(dets_xy).sum(axis=1)
    dists = np.sqrt(-2*mat_mul + np.matrix(tr) + np.matrix(te).T)

    # Subtract the radius from the distances, such that they are < 0 if inside,
    # > 0 if outside.
    dists -= radii

    # Prepend zeros so that argmin is 0 for everything "outside".
    dists = np.hstack([np.zeros((len(scan), 1)), dists])

    return np.argmin(dists, axis=1)

