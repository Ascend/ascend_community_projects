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
import glob
import os
from typing import List, Tuple
import itertools
import numpy as np


def eval_internal(
    dets_xy, dets_cls, dets_inds, gts_xy, gts_inds, ar
):
    a_rad = ar * np.ones(len(gts_inds), dtype=np.float32)

    finds = np.unique(np.r_[dets_inds, gts_inds])

    det_accepted_idxs = defaultdict(list)
    tps = np.zeros(len(finds), dtype=np.uint32)
    fps = np.zeros(len(finds), dtype=np.uint32)
    fns = np.array([np.sum(gts_inds == f) for f in finds], dtype=np.uint32)

    precisions = np.full_like(dets_cls, np.nan)
    recalls = np.full_like(dets_cls, np.nan)
    threshs = np.full_like(dets_cls, np.nan)

    indices = np.argsort(dets_cls, kind="mergesort")  # mergesort for determinism.
    for i, idx in enumerate(reversed(indices)):
        frame = dets_inds[idx]
        iframe = np.where(finds == frame)[0][0]  # Can only be a single one.

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
            mat_mul = np.dot(gts, dets.T)
            te = np.square(gts).sum(axis=1)
            tr = np.square(dets).sum(axis=1)
            c_dists = np.sqrt(np.abs(-2*mat_mul + np.matrix(tr) + np.matrix(te).T))
            not_in_radius = radii[:, None] < c_dists  # -> ngts x ndets, True (=1) if too far, False

            igt, idet = linear_sum_assignment_hungarian(not_in_radius)

            tps[iframe] = np.sum(np.logical_not(not_in_radius[igt, idet]))
            fps[iframe] = (len(dets) - tps[iframe])
            fns[iframe] = len(gts) - tps[iframe]

        tp, fp, fn = np.sum(tps), np.sum(fps), np.sum(fns)
        precisions[i] = tp / (fp + tp) if fp + tp > 0 else np.nan
        recalls[i] = tp / (fn + tp) if fn + tp > 0 else np.nan
    # remove rounding errors
    invalid_recs = np.where(np.diff(recalls))[0]
    for idx in invalid_recs:
        tmp_idx = idx+2
        while recalls[tmp_idx] < recalls[idx]:
            tmp_idx += 1

        # smooth out -- dirty fix
        recalls[idx: tmp_idx+1] = np.linspace(recalls[idx], recalls[tmp_idx], tmp_idx-idx+1)
    ap, peak_f1, eer = _get_ap_prec_rec(recalls, precisions)
    return {
        "precisions": precisions,
        "recalls": recalls,
        "thresholds": threshs,
        "ap": ap,
        "peak_f1": peak_f1,
        "eer": eer,
    }



def linear_sum_assignment_brute_force(
        cost_matrix: np.ndarray,
        maximize: bool = False) -> Tuple[List[int], List[int]]:
    """ maximize:True-> maximum weight matching; False->minimum cost"""
    cost_matrix = np.multiply(np.array(cost_matrix), 1)
    h = cost_matrix.shape[0]
    w = cost_matrix.shape[1]

    if maximize is True:
        cost_matrix = -cost_matrix

    minimum_cost = float("inf")

    if h >= w:
        for i_idices in itertools.permutations(list(range(h)), min(h, w)):
            row_ind = i_idices # non-diag (x,y)s
            col_ind = list(range(w))
            cost = cost_matrix[row_ind, col_ind].sum()
            if cost < minimum_cost:
                minimum_cost = cost
                optimal_row_ind = row_ind
                optimal_col_ind = col_ind
    if h < w:
        for j_idices in itertools.permutations(list(range(w)), min(h, w)):
            row_ind = list(range(h))
            col_ind = j_idices
            cost = cost_matrix[row_ind, col_ind].sum()
            if cost < minimum_cost:
                minimum_cost = cost
                optimal_row_ind = row_ind
                optimal_col_ind = col_ind

    return optimal_row_ind, list(optimal_col_ind)


def linear_sum_assignment_hungarian(cost_matrix, maximize=False):
    """
    Get the element position
    cost_matrix: array of boolean
    if maximize=True, please change to profit_matrix instead
    """

    if cost_matrix.shape[0] == 1 and cost_matrix.shape[1] == 1:
        return [0], [0]
    cost_matrix = np.multiply(np.array(cost_matrix), 1)
    h, w = cost_matrix.shape[0], cost_matrix.shape[1]
    cur_mat = cost_matrix

    #Step 1 - Every column and every row subtract its internal minimum
    zero_count = 0
    if h > w:
        min_dim = w
        while zero_count < min_dim:
            #Step 2 & 3
            ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat)
            zero_count = len(marked_rows) + len(marked_cols)

            if zero_count < min_dim:
                cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)
    else:
        min_dim = h
        while zero_count < min_dim:
            #Step 2 & 3
            ans_pos, marked_rows, marked_cols = mark_matrix(cur_mat, min_w=False)
            zero_count = len(marked_rows) + len(marked_cols)

            if zero_count < min_dim:
                cur_mat = adjust_matrix(cur_mat, marked_rows, marked_cols)
    optimal_row_ind = [x[0] for x in ans_pos]
    optimal_col_ind = [x[1] for x in ans_pos]

    indices = np.argsort(np.array(optimal_row_ind), kind='mergesort')
    optimal_row_ind = np.array(optimal_row_ind)[indices]
    optimal_col_ind = np.array(optimal_col_ind)[indices]
    return optimal_row_ind, optimal_col_ind


def min_zero_row(zero_mat, mark_zero):
    '''
    The function can be splitted into two steps:
    #1 The function is used to find the row which containing the fewest 0.
    #2 Select the zero number on the row, and then marked the element corresponding row and column as False
    '''
    #Find the row
    min_row = [99999, -1]

    for row_num in range(zero_mat.shape[0]):
        if np.sum(zero_mat[row_num] == 1) > 0 and min_row[0] > np.sum(zero_mat[row_num] == 1):
            min_row = [np.sum(zero_mat[row_num] == 1), row_num]

	# Marked the specific row and column as False
    zero_index = np.where(zero_mat[min_row[1]] == 1)[0][0]
    mark_zero.append((min_row[1], zero_index))
    zero_mat[min_row[1], :] = False
    zero_mat[:, zero_index] = False


def min_zero_col(zero_mat, mark_zero):
    '''
    The function can be splitted into two steps:
    #1 The function is used to find the row which containing the fewest 0.
    #2 Select the zero number on the row, and then marked the element corresponding row and column as False
    '''
    #Find the row
    min_col = [99999, -1]

    for col_num in range(zero_mat.shape[1]):
        if np.sum(zero_mat[:, col_num] == 1) > 0 and min_col[0] > np.sum(zero_mat[:, col_num] == 1):
            min_col = [np.sum(zero_mat[:, col_num] == 1), col_num]

	# Marked the specific row and column as False
    zero_index = np.where(zero_mat[:, min_col[1]] == 1)[0][0]
    mark_zero.append((zero_index, min_col[1]))
    zero_mat[:, min_col[1]] = False
    zero_mat[zero_index, :] = False


def mark_matrix(mat, min_w=True):
    '''
    Finding the returning possible solutions for LAP problem.
    '''
    #Transform the matrix to boolean matrix(0 = True, others = False)
    cur_mat = mat
    zero_bool_mat = (cur_mat == 0)
    zero_bool_mat_copy = zero_bool_mat.copy()

    #Recording possible answer positions by marked_zero
    marked_zero = []
    while (True in zero_bool_mat_copy):
        if min_w:
            min_zero_row(zero_bool_mat_copy, marked_zero)
        else:
            min_zero_col(zero_bool_mat_copy, marked_zero)

	#Recording the row and column positions seperately.
    marked_zero_row = []
    marked_zero_col = []
    for i, marked_row in enumerate(marked_zero):
        marked_zero_row.append(marked_zero[i][0])
        marked_zero_col.append(marked_zero[i][1])

    #Step 2-2-1
    non_marked_row = list(set(range(cur_mat.shape[0])) - set(marked_zero_row))

    marked_cols = []
    check_switch = True
    while check_switch:
        check_switch = False
        for i, the_row in enumerate(non_marked_row):
            row_array = zero_bool_mat[the_row, :]
            for j in range(row_array.shape[0]):
                #Step 2-2-2
                if row_array[j] == 1 and j not in marked_cols:
                    #Step 2-2-3
                    marked_cols.append(j)
                    check_switch = True

        for row_num, col_num in marked_zero:
            #Step 2-2-4
            if row_num not in non_marked_row and col_num in marked_cols:
                #Step 2-2-5
                non_marked_row.append(row_num)
                check_switch = True
    #Step 2-2-6
    marked_rows = list(set(range(mat.shape[0])) - set(non_marked_row))

    return marked_zero, marked_rows, marked_cols


def adjust_matrix(mat, cover_rows, cover_cols):
    cur_mat = mat
    non_zero_element = []

    #Step 4-1
    for row_i, cur_row in enumerate(cur_mat):
        if row_i not in cover_rows:
            for col_i, cur_col in enumerate(cur_mat[row_i]):
                if col_i not in cover_cols:
                    non_zero_element.append(cur_mat[row_i][col_i])

    min_num = min(non_zero_element)

    #Step 4-2
    for row_i, cur_row in enumerate(cur_mat):
        if row_i not in cover_rows:
            for col_i, cur_col in enumerate(cur_mat[row_i]):
                if col_i not in cover_cols:
                    cur_mat[row_i, col_i] = cur_mat[row_i, col_i] - min_num
    #Step 4-3
    for row, cover_row in enumerate(cover_rows):
        for col, cover_col in enumerate(cover_cols):
            cur_mat[cover_rows[row], cover_cols[col]] = cur_mat[cover_rows[row], cover_cols[col]] + min_num
    return cur_mat


def _get_ap_prec_rec(rec, prec):
    # make sure the x-input to auc is sorted
    assert np.sum(np.diff(rec) >= 0) == len(rec) - 1
    # compute error matrices
    return _get_auc(rec, prec), _get_peakf1(rec, prec), _get_eer(rec, prec)


def _get_auc(frp, trp):

    x0, y0 = np.ravel(np.asarray(frp).reshape(-1)), np.ravel(np.asarray(trp).reshape(-1))

    if x0.shape[0] < 2:
        raise ValueError(
            "Wrong shape! where x.shape = %s"
            % x0.shape
        )

    direction = 1
    dx = np.diff(x0)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError("x is not in chronological order, please modify : {}.".format(x0))

    area_under_curve = direction * np.trapz(y0, x0)
    if isinstance(area_under_curve, np.memmap):
        area_under_curve = area_under_curve.dtype.type(area_under_curve)
    return area_under_curve



def _get_peakf1(recs, precs):
    return np.max(2 * precs * recs / np.clip(precs + recs, 1e-16, 2 + 1e-16))


def _get_eer(recs, precs):
    # Find the first nonzero or else (0,0) will be the EER :)
    def first_nonzero_element(arr):
        return np.where(arr != 0)[0][0]

    p1 = first_nonzero_element(precs)
    r1 = first_nonzero_element(recs)
    idx = np.argmin(np.abs(precs[p1:] - recs[r1:]))
    return np.average([precs[p1 + idx], recs[r1 + idx]])
