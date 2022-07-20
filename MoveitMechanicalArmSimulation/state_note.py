#!/usr/bin/env python2.7

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, SRI International
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of SRI International nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Acorn Pooley, Mike Lautman

## BEGIN_SUB_TUTORIAL imports
##
## To use the Python MoveIt interfaces, we will import the `moveit_commander`_ namespace.
## This namespace provides us with a `MoveGroupCommander`_ class, a `PlanningSceneInterface`_ class,
## and a `RobotCommander`_ class. More on these below. We also import `rospy`_ and some messages that we will use:
##

import datetime
import threading
import sys
import copy
import signal
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
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

  group_name = "panda_arm"
  move_group = moveit_commander.MoveGroupCommander(group_name)

  ser = serial.Serial('/dev/ttyAMA0', 3000000, 8, 'N', 1) # 'COM7', 3000000, bytesize=8, parity='N', stopbits=1
  flag = ser.is_open

  if flag:
    print('success')
  else:
    print('Open Error')

  time_step = 0
  begin_time = datetime.datetime.now()
  one_step = 2
  
  while True:
    cur_time = (datetime.datetime.now() - begin_time).total_seconds()
    if cur_time >= (time_step + one_step):
      time_step = time_step + one_step
      
      joint_goal = move_group.get_current_joint_values()
      print("joint:", joint_goal[0], joint_goal[1], joint_goal[2], joint_goal[3], joint_goal[4], joint_goal[5], joint_goal[6])
      ser.write("help".encode('utf-8'))
      print("ok?")
    if ISSIGINTUP:
      ser.close()
      print("Exit")
      break 
