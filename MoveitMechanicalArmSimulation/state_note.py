#!/usr/bin/env python2.7
# coding=utf-8

"""
# Copyright(C) 2022. Huawei Technologies Co.,Ltd. All rights reserved.
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
"""


import datetime
import threading
import sys
import copy
import signal
from math import pi
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
import serial


def sigint_handler(signum, frame):
    signum = signum
    frame = frame
    global ISSIGINTUP
    ISSIGINTUP = True
    print("catched interrupt signal")


signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGHUP, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)
ISSIGINTUP = False


if __name__ == '__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('my_move_group_python_interface_tutorial', anonymous=True)

    GROUP_NAME = "panda_arm"
    move_group = moveit_commander.MoveGroupCommander(GROUP_NAME)

    ser = serial.Serial('/dev/ttyAMA0', 3000000, 8, 'N', 1)
    flag = ser.is_open

    if flag:
        print('success')
    else:
        print('Open Error')
        exit()

    time_step = 0
    begin_time = datetime.datetime.now()
    one_step = 2

    while True:
        cur_time = (datetime.datetime.now() - begin_time).total_seconds()
        if cur_time >= (time_step + one_step):
            time_step = time_step + one_step
        
            joint_goal = move_group.get_current_joint_values()
            print("joint:", joint_goal[0], joint_goal[1], joint_goal[2], joint_goal[3], joint_goal[4], joint_goal[5], joint_goal[6])
            # write state to uart dev
            joint_len = len(joint_goal)
            for i in range(joint_len):
                tmp_val = joint_goal[i]
                str = '%.3f'%tmp_val
                ser.write(str.encode('utf-8'))
        if ISSIGINTUP:
            ser.close()
            print("Exit")
            break

