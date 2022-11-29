# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)
# project information
project(Individual)

# Compile options
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)
add_compile_options(-std=c++11 -fPIC -fstack-protector-all -Wall -D_FORTIFY_SOURCE=2 -O2)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -s -pie -pthread")
set(CMAKE_SKIP_RPATH TRUE)

SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

# Header path
include_directories(
    ${MX_SDK_HOME}/include/
    ${MX_SDK_HOME}/opensource/include/
    ${MX_SDK_HOME}/opensource/include/opencv4/
    /home/HwHiAiUser/Ascend/ascend-toolkit/latest/include/
    ./
)

# add host lib path
link_directories(
    ${MX_SDK_HOME}/lib/
    ${MX_SDK_HOME}/lib/modelpostprocessors
    ${MX_SDK_HOME}/opensource/lib/
    ${MX_SDK_HOME}/opensource/lib64/
    /usr/lib/aarch64-linux-gnu/
    /home/HwHiAiUser/Ascend/ascend-toolkit/latest/lib64/
    /usr/local/Ascend/driver/lib64/
    ./
)


aux_source_directory(. sourceList)

add_executable(main ${sourceList})

target_link_libraries(main mxbase opencv_world boost_filesystem glog avformat avcodec avutil cpprest yolov3postprocess ascendcl acl_dvpp_mpi)

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
