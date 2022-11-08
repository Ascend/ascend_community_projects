# Copyright(C) 2021. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
from srcs.detector import Detector
import numpy as np
import onnx
import torch


def main(ckpt_path, datset_name, model_name, panoramic, num_pts, num_scans, onnx_name):
    detector = Detector(
        ckpt_path,
        dataset=datset_name,         # DROW Or JRDB
        model=model_name,          # DROW3 Or DR-SPAAM
        gpu=True,               # Use GPU
        stride=1,               # Optionally downsample scan for faster inference
        panoramic_scan=panoramic    # Set to True if the scan covers 360 degree
    )
    # tell the detector field of view of the LiDAR


    scan = np.random.rand(int(num_pts)) # 450 Or 1091
    scans = np.array([scan for i in range(num_scans)]) # 1 Or 10
    torch.onnx.export(detector.model,
                    detector.return_data(scans),
                    onnx_name,
                    opset_version=11,
                    verbose=True,
                    export_params=True)


    print("Hello onnx")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--ckpt_path",
                        type=str,
                        required=True,
                        help="checkpoints directory.")
    args = parser.parse_args()
    main(os.path.join(args.ckpt_path, "drow_e40.pth"),
        "DROW",
        "DROW3",
        False,
        450,
        1,
        "drow3_e40.onnx")
    main(os.path.join(args.ckpt_path, "dr_spaam_e40.pth"),
        "DROW",
        "DR-SPAAM",
        False,
        450,
        10,
        "dr_spaam_e40.onnx")
    main(os.path.join(args.ckpt_path, "ckpt_jrdb_ann_ft_drow3_e40.pth"),
        "JRDB",
        "DROW3",
        True,
        1091,
        1,
        "drow3_jrdb_e40.onnx")
    main(os.path.join(args.ckpt_path, "ckpt_jrdb_ann_ft_dr_spaam_e20.pth"),
        "JRDB",
        "DR-SPAAM",
        True,
        1091,
        10,
        "dr_spaam_jrdb_e20.onnx")
