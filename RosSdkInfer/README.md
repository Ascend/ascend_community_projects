# RosSdkInfer

## 1 介绍

本开发项目在Atlas200dk上基于ROS系统实现MindxSDK推理，实现ROS节点通讯，一个节点为sdk的server端，初始化sdk的pipeline,并且等待请求，另一个节点实现图片发送，并且等待sdk的推理结果。本样使用yolov3模型实现对图片的常见物品检测以及其定位。启动sdk-server端的节点等待其他节点的request,启动client端，向server端发送一图片；server端把图片推理结果再发送到client端，client端把结果（物体的对角线以及类别）用opencv画于图片中并且保存。

1.环境依赖
2.200dk-conda-sdk环境搭建;
3.基于200dk-ROS系统移植;
4.ROS工程创建;
5.项目文件部署;
6.运行;


## 1 环境依赖

* 支持的硬件形态和操作系统版本

  | 硬件形态                              | 操作系统版本   |
  | ------------------------------------- | -------------- |
  | x86+Atlas 300I 推理卡（型号3010）  | Ubuntu 18.04.1 |
  | ARM+Atlas 300I 推理卡 （型号3000）    | Ubuntu 18.04.1 |
  | aarch+Atlas 200dk 推理卡（型号3010）  | Ubuntu 18.04.1 |
  | x86+Atlas 200dk 推理卡（型号3000）  | Ubuntu 18.04.1 |

* 软件依赖

  | 软件名称 | 版本  |
  | -------- | ----- |
  | cmake    | 3.5.+ |
  | mxVision | 2.0.4 |
  | Python   | 3.9.2 |
  | CANN   | 5.0.4 |
  | OpenCV   | 4.5.3 |
  | gcc      | 7.5.0 |
  | ROS      | melodic |

## 2 200dk-conda-sdk环境搭建

关于200dk-conda-sdk环境的搭建的详细说明可以参考：https://gitee.com/ascend/docs-openmind/blob/master/guide/mindx/ascend_community_projects/tutorials/200dk%E5%BC%80%E5%8F%91%E6%9D%BF%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md

## 3 基于200dk-ROS系统移植

基于200dk-ROS系统的移植可以参考：https://gitee.com/huang-heyang/docs-openmind/blob/master/guide/mindx/ascend_community_projects/tutorials/200dk-ROS%E7%B3%BB%E7%BB%9F%E7%A7%BB%E6%A4%8D.md

## 4 ROS工程创建

1、创建工作空间

$mkdir -p ~/catkin_ws/src

$cd ~/catkin_ws/src

$catkin_init_workspace

2、编译工作空间

$cd ~/catkin_ws

$catkin_make

3、设置环境变量

$source devel/setup.bash

4、检测环境变量

$echo $ROS_PACKAGE_PATH

5、创建功能包

$cd ~/catkin_ws/src

$catkin_create_pkg test_pkg

6、编译功能包

$cd ~/catkin_ws

$catkin_make

7、创建工程

$cd ~/catkin_ws/src

$catkin_create_pkg test_pkg roscpp rospy std_msgs geometry_msgs turtlesim

## 5 项目文件部署

1、删除test_pkg下的文件

$rm CMakeLists.txt package.xml src include -rf

2、把本项目test_pkg下的文件拷贝过去

3、执行catkin_make（这一步主要是生成自定义数据代码文件）

## 6 运行

1、运行前先输入：

pip install pyyaml

pip install rospkg

pip install pydot

source ~/catkin_ws/devel/setup.bash

2、文件修改：

修改my_py_test_server.py中pipeline的路径

模型文件放在scripts/下，根据模型的实际情况，修改pipeline内容里面的路径

准备好jpg格式的图片放在任意目录下，并且修改my_py_test_server.py中的路径

修改my_py_test_client.py中的图片的读取路径已经图片的保存路径

3、运行

使用三个终端窗口，第一个启动roscore,第二个启动rosrun test_pkg my_py_test_server.py,第三个启动rosrun test_pkg my_py_test_client.py

最终会得到推理结果图片，并保存于本地。

![输入图片说明](../figures/result.jpg "result.jpg")
