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
import math
import torch
import numpy as np

from srcs.nets.drow_net import DrowNet
from srcs.nets.dr_spaam import DrSpaam

_PI = np.pi

__all__ = ["Detector", "scans_to_cutout"]


class Detector(object):
    def __init__(
        self, ckpt_file, dataset="JRDB", model="DROW3", gpu=True, stride=1, panoramic_scan=False
    ):
        """A warpper class around DROW3 or DR-SPAAM network for end-to-end inference.

        Args:
            ckpt_file (str): Path to checkpoint
            model (str): Model name, "DROW3" or "DR-SPAAM".
            gpu (bool): True to use GPU. Defaults to True.
            stride (int): Downsample scans for faster inference.
            panoramic_scan (bool): True if the scan covers 360 degree.
        """
        self._device = gpu
        self._stride = stride
        self._use_dr_spaam = model == "DR-SPAAM"

        self._scan_phi = None
        self.data = np.random.rand(56)

        if "jrdb" in dataset or "JRDB" in dataset:
            self._laser_fov_deg = 360
        elif "drow" in dataset or "DROW" in dataset:
            self._laser_fov_deg = 225
        else:
            raise NotImplementedError(
                "Received unsupported dataset {}, break.".format(
                    dataset
                )
            )
        if model == "DROW3":
            self._model = DrowNet(
                dropout=0.5, cls_loss=None)
        elif model == "DR-SPAAM":
            self._model = DrSpaam(
                dropout=0.5,
                num_pts=56,
                embedding_length=128,
                alpha=0.5,
                window_size=17,
                panoramic_scan=panoramic_scan,
                cls_loss=None,
            )
        else:
            raise NotImplementedError(
                "Received unsupported model {}, break.".format(
                    model
                )
            )

        ckpt = torch.load(ckpt_file)
        self._model.load_state_dict(ckpt["model_state"])
        self.model = self._model
        self._model.eval()
        if gpu:
            torch.backends.cudnn.benchmark = True
            self._model = self._model.cuda()
        

    def return_data(self, scan):
        bisec_fov_rad = 0.5 * np.deg2rad(self._laser_fov_deg)
        self._scan_phi = np.linspace(
            -bisec_fov_rad, bisec_fov_rad, max(scan.shape), dtype=np.float32
        )

        # preprocess
        ct = scans_to_cutout(
            scan,
            self._scan_phi,
            stride=self._stride,
            win_size=[1.0, 0.5],
            num_cutout_pts=56,
            padding_val=29.99,
        )
        ct = torch.from_numpy(ct).float()

        if self._device:
            ct = ct.cuda()
        return ct.unsqueeze(dim=0)


def scans_to_cutout(
        scans,
        scan_phi,
        stride=1,
        win_size=None,
        num_cutout_pts=56,
        padding_val=29.99,
):

    num_scans, num_pts = scans.shape # (1, num_pts)

    # size (width) of the window
    pt_dists = scans[:, ::stride]
    bisec_alp = np.arctan(0.5 * win_size[0] / np.maximum(pt_dists, 1e-2))
    scan_diff = scan_phi[1] - scan_phi[0]
    # cutout indices
    del_alp = 2.0 * bisec_alp / (num_cutout_pts - 1)
    ang_cutout = (
            scan_phi[::stride]
            - bisec_alp
            + np.arange(num_cutout_pts).reshape(num_cutout_pts, 1, 1) * del_alp
    )
    ang_cutout = (ang_cutout + _PI) % (2.0 * _PI) - _PI  # warp angle
    inds_cutout = (ang_cutout - scan_phi[0]) / scan_diff
    outter_mask = np.logical_or(inds_cutout < 0, inds_cutout > num_pts - 1)

    # cutout (linear interp)
    inds_cutout_lower = np.core.umath.clip(np.floor(inds_cutout), 0, num_pts - 1).astype(np.int32)
    inds_cutout_upper = np.core.umath.clip(inds_cutout_lower + 1, 0, num_pts - 1).astype(np.int32)
    inds_ct_ratio = np.core.umath.clip(inds_cutout - inds_cutout_lower, 0.0, 1.0)
        
    inds_offset = (np.arange(num_scans).reshape(1, num_scans, 1) * num_pts)
    cutout_lower = np.take(scans, inds_cutout_lower + inds_offset)
    cutout_upper = np.take(scans, inds_cutout_upper + inds_offset)
    ct0 = cutout_lower + inds_ct_ratio * (cutout_upper - cutout_lower)

    # use area sampling for down-sampling (close points)
    if True:
        num_pts_win = inds_cutout[-1] - inds_cutout[0]
        area_mask = num_pts_win > num_cutout_pts
        if np.sum(area_mask) > 0:
            # sample the window with more points than the actual number of points
            sample_area = int(math.ceil(np.max(num_pts_win) / num_cutout_pts))
            num_ct_pts_area = sample_area * num_cutout_pts
            del_alp_area = 2.0 * bisec_alp / (num_ct_pts_area - 1)
            ang_ct_area = (
                    scan_phi[::stride]
                    - bisec_alp
                    + np.arange(num_ct_pts_area).reshape(num_ct_pts_area, 1, 1)
                    * del_alp_area
            )
            ang_ct_area = (ang_ct_area + _PI) % (2.0 * _PI) - _PI  # warp angle
            inds_ct_area = (ang_ct_area - scan_phi[0]) / scan_diff
            inds_ct_area = np.rint(np.core.umath.clip(inds_ct_area, 0, num_pts - 1)).astype(np.int32)
            ct_area = np.take(scans, inds_ct_area + inds_offset)
            ct_area = ct_area.reshape(
                    num_cutout_pts, sample_area, num_scans, pt_dists.shape[1]
                ).mean(axis=1)
            ct0[:, area_mask] = ct_area[:, area_mask]

    # normalize cutout
    ct0[outter_mask] = padding_val
    ct1 = np.core.umath.clip(ct0, pt_dists - win_size[1], pt_dists + win_size[1])
    ct1 = (ct1 - pt_dists) / win_size[1]
    cutout_scan = np.ascontiguousarray(ct1.transpose((2, 1, 0)), dtype=np.float32)

    return cutout_scan
