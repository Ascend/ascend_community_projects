cmake_minimum_required(VERSION 3.5.2)
project(AlphaposePostProcessPlugin)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)

set(PLUGIN_NAME "mxpi_alphaposepostprocess")
set(TARGET_LIBRARY ${PLUGIN_NAME})
set(MX_SDK_HOME "$ENV{MX_SDK_HOME}")

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${MX_SDK_HOME}/include)
include_directories(${MX_SDK_HOME}/opensource/include)
include_directories(${MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories(${MX_SDK_HOME}/opensource/include/glib-2.0)
include_directories(${MX_SDK_HOME}/opensource/lib/glib-2.0/include)
include_directories(${MX_SDK_HOME}/opensource/include/opencv4)
include_directories(${PROJECT_SOURCE_DIR}/../../proto)
link_directories(${MX_SDK_HOME}/lib)
link_directories(${MX_SDK_HOME}/opensource/lib)
link_directories(${PROJECT_SOURCE_DIR}/../../proto/lib)

add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")
add_definitions(-DENABLE_DVPP_INTERFACE)
add_library(${TARGET_LIBRARY} SHARED MxpiAlphaposePostProcess.cpp)
target_link_libraries(${TARGET_LIBRARY} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0 gmodule-2.0 glog)
target_link_libraries(${TARGET_LIBRARY} mxpidatatype plugintoolkit mxbase mindxsdk_protobuf mxpialphaposeproto)
