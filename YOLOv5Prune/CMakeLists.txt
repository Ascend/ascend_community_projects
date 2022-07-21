# Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.10.1)

# project information
project(MindX_SDK_Sample)   

set(MX_SDK_HOME $ENV{MX_SDK_HOME})      

if (NOT DEFINED ENV{MX_SDK_HOME})      
    string(REGEX REPLACE "(.*)/(.*)/(.*)/(.*)" "\\1" MX_SDK_HOME  ${CMAKE_CURRENT_SOURCE_DIR})
    message(STATUS "set default MX_SDK_HOME: ${MX_SDK_HOME}")
else ()
    message(STATUS "env MX_SDK_HOME: ${MX_SDK_HOME}")
endif()

# Compile options
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)  
add_definitions(-Dgoogle=mindxsdk_private)      
add_compile_options(-std=c++11 -fPIC -fstack-protector-all -Wall)   

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})       
set(CMAKE_CXX_FLAGS_DEBUG "-g")                                      
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -pie")

# Header path  
include_directories(
    ${MX_SDK_HOME}/include/     
    ${MX_SDK_HOME}/opensource/include/
    ${MX_SDK_HOME}/opensource/include/opencv4
)

# add host lib path   
link_directories(
    ${MX_SDK_HOME}/lib/
    ${MX_SDK_HOME}/lib/modelpostprocessors/
    ${MX_SDK_HOME}/opensource/lib/
    ${MX_SDK_HOME}/opensource/lib64/
)

add_executable(main main_sdk.cpp)

target_link_libraries(main glog mxbase plugintoolkit mxpidatatype streammanager cpprest mindxsdk_protobuf opencv_world)

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})





