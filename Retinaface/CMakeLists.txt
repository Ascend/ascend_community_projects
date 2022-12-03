cmake_minimum_required(VERSION 3.10)
project(retinaface)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

include_directories(./RetinafacePostProcess)
include_directories(./RetinafaceDetection)
include_directories(/usr/local/Ascend/ascend-toolkit/latest/arm64-linux/runtime/include/)
file(GLOB_RECURSE Retinaface_POSTPROCESS ${PROJECT_SOURCE_DIR}/RetinafacePostProcess/RetinafacePostProcess.cpp)
file(GLOB_RECURSE Retinaface_DETECTION ${PROJECT_SOURCE_DIR}/RetinafaceDetection/RetinafaceDetection.cpp)
set(TARGET retinaface)
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

add_executable(retinaface main.cpp ${Retinaface_DETECTION} ${Retinaface_POSTPROCESS})
target_link_libraries(retinaface
        glog
        mxbase
        cpprest
        opencv_world
        boost_filesystem
        )
