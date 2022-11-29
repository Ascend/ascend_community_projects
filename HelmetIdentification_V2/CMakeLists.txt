# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
  
# 最低CMake版本
cmake_minimum_required(VERSION 3.5.1)

# 项目名
project(HelmetIdentification)

# 配置环境变量MX_SDK_HOME，如：/home/xxxxxxx/MindX_SDK/mxVision,可在远程环境中用指令env查看
set(MX_SDK_HOME $ENV{MX_SDK_HOME})

if (NOT DEFINED ENV{MX_SDK_HOME})
    set(MX_SDK_HOME "/usr/local/Ascend/mindx_sdk")
    message(STATUS "set default MX_SDK_HOME: ${MX_SDK_HOME}")
else ()
    message(STATUS "env MX_SDK_HOME: ${MX_SDK_HOME}")
endif()

add_subdirectory("./src")