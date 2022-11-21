# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(MonocularDepthEstimation)

set(MX_SDK_HOME $ENV{MX_SDK_HOME})

if (NOT DEFINED ENV{MX_SDK_HOME})
    set(MX_SDK_HOME "/usr/local/Ascend/mindx_sdk")
    message(STATUS "set default MX_SDK_HOME: ${MX_SDK_HOME}")
else ()
    message(STATUS "env MX_SDK_HOME: ${MX_SDK_HOME}")
endif()

add_subdirectory("./src")
