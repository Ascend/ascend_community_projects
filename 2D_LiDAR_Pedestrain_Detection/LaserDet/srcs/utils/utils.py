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
import numpy as np



def xy_to_rphi(x, y):
    return np.hypot(x, y), np.arctan2(y, x)



def global_to_canonical(scan_r, scan_phi, dets_r, dets_phi):
    dx = np.sin(dets_phi - scan_phi) * dets_r
    dy = np.cos(dets_phi - scan_phi) * dets_r - scan_r
    return dx, dy


def rphi_to_xy(r, phi):
    return r * np.cos(phi), r * np.sin(phi)


def canonical_to_global(scan_r, scan_phi, dx, dy):
    y_shift = scan_r + dy
    phi_shift = np.arctan2(
        dx, y_shift
    )
    dets_phi = phi_shift + scan_phi
    dets_r = y_shift / np.cos(phi_shift)
    return dets_r, dets_phi



def trim_the_scans(
    scans,
    scan_phi,
    stride=1,
    win_width=1.66,
    win_depth=1.0,
    num_ct_pts=48,
    pad_val=29.99,
):
    _pi = np.pi
    num_scans, num_pts = scans.shape

    bisec_alp = np.arctan(0.5 * win_width / np.maximum(scans[:, ::stride], 1e-2))

    # delta alpha
    del_alp = 2.0 * bisec_alp / (num_ct_pts - 1)
    ang_cutout = (
            scan_phi[::stride]
            - bisec_alp
            + np.arange(num_ct_pts).reshape(num_ct_pts, 1, 1) * del_alp
    )
    ang_cutout = (ang_cutout + _pi) % (2.0 * _pi) - _pi
    inds_cutout = (ang_cutout - scan_phi[0]) / (scan_phi[1] - scan_phi[0])
    outter_mask = np.logical_or(inds_cutout < 0, inds_cutout > num_pts - 1)

    # cutout (linear interp)
    inds_cutout_lower = np.core.umath.clip(np.floor(inds_cutout), 0, num_pts - 1).astype(np.int32)
    inds_cutout_upper = np.core.umath.clip(inds_cutout_lower + 1, 0, num_pts - 1).astype(np.int32)
    inds_ct_ratio = np.core.umath.clip(inds_cutout - inds_cutout_lower, 0.0, 1.0)

    inds_offset = (np.arange(num_scans).reshape(1, num_scans, 1) * num_pts)
    cutout_lower = np.take(scans, inds_cutout_lower + inds_offset)
    cutout_upper = np.take(scans, inds_cutout_upper + inds_offset)
    ct0 = cutout_lower + inds_ct_ratio * (cutout_upper - cutout_lower)


    pt_dists = scans[:, ::stride]
    num_pts_win = inds_cutout[-1] - inds_cutout[0]
    area_mask = num_pts_win > num_ct_pts
    if np.sum(area_mask) > 0:
        sample_area = int(math.ceil(np.max(num_pts_win) / num_ct_pts))
        num_ct_pts_area = sample_area * num_ct_pts
        del_alp_area = 2.0 * bisec_alp / (num_ct_pts_area - 1)
        ang_ct_area = (
                    scan_phi[::stride]
                    - bisec_alp
                    + np.arange(num_ct_pts_area).reshape(num_ct_pts_area, 1, 1)
                    * del_alp_area
        )
        ang_ct_area = (ang_ct_area + _pi) % (2.0 * _pi) - _pi  # warp angle
        inds_ct_area = (ang_ct_area - scan_phi[0]) / (scan_phi[1] - scan_phi[0])
        inds_ct_area = np.rint(np.core.umath.clip(inds_ct_area, 0, num_pts - 1)).astype(np.int32)
        ct_area = np.take(scans, inds_ct_area + inds_offset)
        ct_area = ct_area.reshape(
                    num_ct_pts, sample_area, num_scans, pt_dists.shape[1]
                ).mean(axis=1)
        ct0[:, area_mask] = ct_area[:, area_mask]


    ct0[outter_mask] = pad_val
    ct1 = np.core.umath.clip(ct0, pt_dists - win_depth, pt_dists + win_depth)
    ct1 = (ct1 - pt_dists) / win_depth
    cutout_scan = np.ascontiguousarray(ct1.transpose((2, 1, 0)), dtype=np.float32)

    return cutout_scan
