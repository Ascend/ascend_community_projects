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
import torch
import torch.nn as nn
import torch.nn.functional as Func


class DrowNet(nn.Module):
    def __init__(self, dropout=0.5, cls_loss=None):
        super(DrowNet, self).__init__()

        self.dropout = dropout

        self.conv_block_1 = nn.Sequential(
            self._conv_1d(1, 64), self._conv_1d(64, 64), self._conv_1d(64, 128)
        )
        self.conv_block_2 = nn.Sequential(
            self._conv_1d(128, 128), self._conv_1d(128, 128), self._conv_1d(128, 256)
        )
        self.conv_block_3 = nn.Sequential(
            self._conv_1d(256, 256), self._conv_1d(256, 256), self._conv_1d(256, 512)
        )
        self.conv_block_4 = nn.Sequential(self._conv_1d(512, 256), self._conv_1d(256, 128))

        self.conv_cls = nn.Conv1d(128, 1, kernel_size=1)
        self.conv_reg = nn.Conv1d(128, 2, kernel_size=1)

        # classification loss
        self.cls_loss = (
            cls_loss if cls_loss is not None else Func.binary_cross_entropy_with_logits
        )

        # initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity="leaky_relu")
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, signal):

        num_bs, num_ct, num_sc, num_pt = signal.shape

        # start with scans
        flow = signal.view(-1, 1, num_pt)
        flow = self._conv_et_pool_1d(flow, self.conv_block_1)  
        flow = self._conv_et_pool_1d(flow, self.conv_block_2)  

        flow = flow.view(num_bs, num_ct, num_sc, flow.shape[-2], flow.shape[-1])
        flow = torch.sum(flow, dim=2)  

        # forward fused cutout
        flow = flow.view(num_bs * num_ct, flow.shape[-2], flow.shape[-1])
        flow = self._conv_et_pool_1d(flow, self.conv_block_3)  
        flow = self.conv_block_4(flow)
        
        flow = Func.avg_pool1d(flow, kernel_size=7)  

        pred_cls = self.conv_cls(flow).view(num_bs, num_ct, -1)  
        pred_reg = self.conv_reg(flow).view(num_bs, num_ct, 2)  

        return pred_cls, pred_reg

    def _conv_et_pool_1d(self, signal, convblock):
        flow = convblock(signal)
        flow = Func.max_pool1d(flow, kernel_size=2)
        if self.dropout > 0:
            flow = Func.dropout(flow, p=self.dropout, training=self.training)

        return flow

    def _conv_1d(self, in_channel, out_channel):
        return nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
