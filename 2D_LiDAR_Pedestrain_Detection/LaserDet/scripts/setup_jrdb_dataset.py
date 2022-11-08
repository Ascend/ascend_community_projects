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
import json
import os
import stat
import shutil
from collections import defaultdict
import numpy as np
import rosbag as rb

FLAGS = os.O_WRONLY | os.O_CREAT 
MODES = stat.S_IWUSR | stat.S_IRUSR  
# Set root dir to JRDB
_JRDB_DIR = "./dataset/JRDB"


def bag_to_txt(split):
    # .bag from rosbag to .txt for lasers
    jrdb_dir = os.path.join(_JRDB_DIR, split + "_dataset")

    timestamps_dir = os.path.join(jrdb_dir, "timestamps")
    rosbag_dir = timestamps_dir.replace("timestamps", "rosbags")
    to_laser = timestamps_dir.replace("timestamps", "laser")
    seq_names = os.listdir(timestamps_dir)

    
    if os.path.exists(to_laser):
        shutil.rmtree(to_laser)
    os.mkdir(to_laser)

    for idx, seq_name in enumerate(seq_names):
        seq_laser_dir = os.path.join(to_laser, seq_name)
        os.mkdir(seq_laser_dir)
        
        ros_bag = rb.Bag(os.path.join(rosbag_dir, seq_name + ".bag"))

        # unroll your laser messages
        timestamp_list = []
        for cnter, (_, msg, timer) in enumerate(
            ros_bag.read_messages(topics=["segway/scan_multi"])
        ):
            
            laser_file = str(cnter).zfill(6) + ".txt"
            np.savetxt(
                os.path.join(seq_laser_dir, laser_file), 
                np.array(msg.ranges), 
                newline=" "
            )

            timestamp_list.append(timer.to_sec())

        np.savetxt(
            os.path.join(seq_laser_dir, "timestamps.txt"),
            np.array(timestamp_list),
            newline=" ",
        )

        ros_bag.close()


def synchronize_pcl_img_laser(split):
    seq_names = os.listdir(
        os.path.join(_JRDB_DIR, split + "_dataset", "timestamps")
    )
    for idx, seq_name in enumerate(seq_names):
        jrdb_dir = os.path.join(_JRDB_DIR, split + "_dataset")

        timestamps_dir = os.path.join(jrdb_dir, "timestamps", seq_name)
        lasers_dir = timestamps_dir.replace("timestamps", "lasers")

        # loading pointcloud frames
        with open(timestamps_dir+"/"+"frames_pc.json", "r") as f:
            pc_data_load = json.load(f)["data"]
        # loading image frames
        with open(timestamps_dir+"/"+"frames_img.json", "r") as f:
            im_data_load = json.load(f)["data"]

        # matching pc and im frame
        pc_tim = np.array([float(f["timestamp"]) for f in pc_data_load]).reshape(-1, 1)
        im_tim = np.array([float(f["timestamp"]) for f in im_data_load]).reshape(1, -1)

        p_i_inds = np.abs(pc_tim - im_tim).argmin(axis=1)

        # matching pc and laser
        laser_tim = np.loadtxt(
            os.path.join(lasers_dir, "timestamps.txt"), dtype=np.float64
        )
        
        p_l_inds = np.abs(pc_tim - laser_tim.reshape(1, -1)).argmin(axis=1)

        # create a merged frame dict
        frames_collect = []
        for i, pc_frame in enumerate(pc_data_load):
            merged_frame = defaultdict({
                "pc_frame": pc_data_load[i],
                "im_frame": im_data_load[p_i_inds[i]],
                "laser_frame": {
                    "url": os.path.join(
                        "laser",
                        seq_name,
                        str(p_l_inds[i]).zfill(6) + ".txt",
                    ),
                    "name": "laser_combined",
                    "timestamp": laser_tim[p_l_inds[i]],
                },
                "timestamp": pc_data_load[i]["timestamp"],
                "frame_id": pc_data_load[i]["frame_id"],
            })

            # correct file url for pc and im
            for pc_dict in merged_frame["pc_frame"]["pointclouds"]:
                f_name = os.path.basename(pc_dict["url"])
                pc_dict["url"] = os.path.join(
                    "pointclouds", pc_dict["name"], seq_name, f_name
                )

            for im_dict in merged_frame["im_frame"]["cameras"]:
                f_name = os.path.basename(im_dict["url"])
                cam_name = (
                    "image_stitched"
                    if im_dict["name"] == "stitched_image0"
                    else im_dict["name"][:-1] + "_" + im_dict["name"][-1]
                )
                im_dict["url"] = os.path.join("images", cam_name, seq_name, f_name)

            frames_collect.append(merged_frame)

        # write to file
        write_json = {"data": frames_collect}
        json_file = os.path.join(timestamps_dir, "frames_pc_im_laser.json")
        with os.fdopen(os.open(json_file, FLAGS, MODES), "w") as fp:
            json.dump(write_json, fp)
        


if __name__ == "__main__":
    print("Setting up JRDB dataset...")
    for spl in ["train", "test"]:
        bag_to_txt(spl)
        synchronize_pcl_img_laser(spl)
