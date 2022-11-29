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
from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F


class DrSpaam(nn.Module):
    def __init__(
        self,
        dropout=0.5,
        num_pts=48,
        alpha=0.5,
        embed_len=128,
        win_size=7,
        pano_scan=False,
        cls_loss=None,
    ):
        super(DrSpaam, self).__init__()

        self.dropout = dropout

        # backbone
        self.conv_u1 = nn.Sequential(
            self._conv_1d(1, 64), self._conv_1d(64, 64), self._conv_1d(64, 128)
        )
        self.conv_u2 = nn.Sequential(
            self._conv_1d(128, 128), self._conv_1d(128, 128), self._conv_1d(128, 256)
        )
        self.conv_u3 = nn.Sequential(
            self._conv_1d(256, 256), self._conv_1d(256, 256), self._conv_1d(256, 512)
        )
        self.conv_u4 = nn.Sequential(self._conv_1d(512, 256), self._conv_1d(256, 128))

        # detection layer
        self.head_cls = nn.Conv1d(128, 1, kernel_size=1)
        self.head_reg = nn.Conv1d(128, 2, kernel_size=1)

        # spatial attention
        self.spatial_attention_memory = _SpatialAttentionMemory(
            n_pts=int(ceil(num_pts / 4)),
            n_channel=256,
            embed_len=embed_len,
            alpha=alpha,
            win_size=win_size,
            pano_scan=pano_scan,
        )

        # initialize weights
        for idx, module in enumerate(self.modules()):
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, signal, inference=False):

        n_bs, n_cutout, n_scan, n_pc = signal.shape

        if not inference:
            self.spatial_attention_memory.reset()

        # process scan sequentially
        n_scan = signal.shape[2]
        for i in range(n_scan):
            signal_i = signal[:, :, i, :]  # (bs, cutout, pc)

            # extract feature from current scan
            flow = signal_i.view(n_bs * n_cutout, 1, n_pc)
            flow = self._conv_et_pool_1d(flow, self.conv_u1)  # /2
            flow = self._conv_et_pool_1d(flow, self.conv_u2)  # /4
            flow = flow.view(n_bs, n_cutout, flow.shape[-2], flow.shape[-1])  # (bs, cutout, C, pc)

            # combine current feature with memory
            flow, sim_score = self.spatial_attention_memory(flow)  # (bs, cutout, C, pc)

        # detection using combined feature memory
        flow = flow.view(n_bs * n_cutout, flow.shape[-2], flow.shape[-1])
        flow = self._conv_et_pool_1d(flow, self.conv_u3)  # /8
        flow = self.conv_u4(flow)
        flow = F.avg_pool1d(flow, kernel_size=flow.shape[-1])  # (bs * cutout, C, 1)

        pred_cls = self.head_cls(flow).view(n_bs, n_cutout, -1)  # (bs, cutout, cls)
        pred_reg = self.head_reg(flow).view(n_bs, n_cutout, 2)  # (bs, cutout, 2)

        return pred_cls, pred_reg, sim_score

    def _conv_et_pool_1d(self, signal, conv_block):
        flow = conv_block(signal)
        flow = F.max_pool1d(flow, kernel_size=2)
        if self.dropout > 0:
            flow = F.dropout(flow, p=self.dropout, training=self.training)

        return flow

    def _conv_1d(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )    


class _SpatialAttentionMemory(nn.Module):
    def __init__(
        self, n_pts, n_channel, embed_len, alpha, win_size, pano_scan
    ):
        
        super(_SpatialAttentionMemory, self).__init__()
        self._alpha = alpha
        self._win_size = win_size
        self._embed_len = embed_len
        self._pano_scan = pano_scan

        self.atten_mem = None
        self.neighbour_masks = None
        self.neighbour_inds = None

        self.custom_conv = nn.Sequential(
            nn.Conv1d(n_channel, self._embed_len, kernel_size=n_pts, padding=0),
            nn.BatchNorm1d(self._embed_len),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        for idx, module in enumerate(self.modules()):
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def reset(self):
        self.atten_mem = None

    def forward(self, sig_new):
        if self.atten_mem is None:
            self.atten_mem = sig_new
            return self.atten_mem, None

        n_batch, n_cutout, n_channel, n_pts = sig_new.shape

        
        self.neighbour_masks, self.neighbour_inds = self._generate_neighbour_masks(
            sig_new
            )

        embed_x = self.custom_conv(
            sig_new.view(-1, n_channel, n_pts)
            ).view(-1, n_cutout, self._embed_len)

        
        embed_tmp = self.custom_conv(
            self.atten_mem.view(-1, n_channel, n_pts)
            ).view(-1, n_cutout, self._embed_len)

        # pair-wise similarity (batch, cutout, cutout)
        sim_score = torch.matmul(embed_x, embed_tmp.permute(0, 2, 1))

        # masked softmax
        sim_score = sim_score - 1e10 * (1.0 - self.neighbour_masks)
        max_sim = sim_score.max(dim=-1, keepdim=True)[0]
        exp_sim = torch.exp(sim_score - max_sim) * self.neighbour_masks
        exps_sum = exp_sim.sum(dim=-1, keepdim=True)
        sim_score = exp_sim / exps_sum

        # weighted average on the template
        atten_mem = self.atten_mem.view(n_batch, n_cutout, -1)
        atten_mem_w = torch.matmul(sim_score, atten_mem).view(-1, n_cutout, n_channel, n_pts)

        # update memory using auto-regressive
        self.atten_mem = self._alpha * sig_new + (1.0 - self._alpha) * atten_mem_w

        return self.atten_mem, sim_score

    def _generate_neighbour_masks(self, sig):
        
        n_cutout = sig.shape[1]
        half_ws = int(self._win_size / 2)
        inds_col = torch.arange(n_cutout).unsqueeze(dim=-1).long()
        win_inds = torch.arange(-half_ws, half_ws + 1).long()
        inds_col = inds_col + win_inds.unsqueeze(dim=0)  # (cutout, neighbours)
        
        inds_col = (
            inds_col % n_cutout
            if self._pano_scan and not self.training
            else inds_col.clamp(min=0, max=n_cutout - 1)
        )
        inds_row = torch.arange(n_cutout).unsqueeze(dim=-1).expand_as(inds_col).long()
        inds_full = torch.stack((inds_row, inds_col), dim=2).view(-1, 2)

        nb_masks = torch.zeros(n_cutout, n_cutout).float()
        nb_masks[inds_full[:, 0], inds_full[:, 1]] = 1.0
        return nb_masks.cuda(sig.get_device()) if sig.is_cuda else nb_masks, inds_full
