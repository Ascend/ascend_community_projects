cmake_minimum_required(VERSION 3.10)
project(baserefinedet)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(./RefineDetPostProcess)
include_directories(./RefineDetDetection)
file(GLOB_RECURSE REFINEDET_POSTPROCESS ${PROJECT_SOURCE_DIR}/RefineDetPostProcess/*cpp)
file(GLOB_RECURSE REFINEDET_DETECTION ${PROJECT_SOURCE_DIR}/RefineDetDetection/*cpp)
set(TARGET refinedet)
add_compile_options(-std=c++11 -fPIE -fstack-protector-all -fPIC -Wl,-z,relro,-z,now,-z,noexecstack -s -pie -Wall)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(MX_SDK_HOME "$ENV{MX_SDK_HOME}")

include_directories(
        ${MX_SDK_HOME}/include
        ${MX_SDK_HOME}/opensource/include
        ${MX_SDK_HOME}/opensource/include/opencv4
        ${MX_SDK_HOME}/include/MxBase/postprocess/include
        /usr/local/Ascend/ascend-toolkit/latest/include
)

link_directories(
        ${MX_SDK_HOME}/lib
        ${MX_SDK_HOME}/opensource/lib
        ${MX_SDK_HOME}/lib/modelpostprocessors
        /usr/local/Ascend/ascend-toolkit/latest/acllib/lib64
        /usr/local/Ascend/driver/lib64/
)

add_executable(refinedet main.cpp ${REFINEDET_DETECTION} ${REFINEDET_POSTPROCESS})
target_link_libraries(refinedet
        glog
        mxbase
        cpprest
        opencv_world
        )
