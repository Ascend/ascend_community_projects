
cmake_minimum_required(VERSION 3.5.2)
project(deepsort)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)
set(PLUGIN_NAME "deepsort")
set(TARGET_LIBRARY ${PLUGIN_NAME})

set(MX_SDK_HOME "/home/wangshengke1/MindX_SDK/mxVision")

include_directories(/usr/include/eigen3)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${MX_SDK_HOME}/include)
include_directories(${MX_SDK_HOME}/opensource/include)
include_directories(${MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories(${MX_SDK_HOME}/opensource/include/glib-2.0)
include_directories(${MX_SDK_HOME}/opensource/lib/glib-2.0/include)

include_directories(/home/wangshengke1/MindX_SDK/mxVision/opensource/include/opencv4)


link_directories(${MX_SDK_HOME}/lib)
link_directories(${MX_SDK_HOME}/opensource/lib)

add_compile_options(-std=c++11 -fPIC -fstack-protector-all -pie -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")

add_definitions(-DENABLE_DVPP_INTERFACE)

add_library(${TARGET_LIBRARY} SHARED 
    DeepSort.cpp 
    kalmanfilter.cpp 
    linear_assignment.cpp 
    nn_matching.cpp 
    track.cpp 
    tracker.cpp 
    ../DeepAppearanceDescriptor/model.cpp 
    ../MunkresAssignment/hungarianoper.cpp 
    ../MunkresAssignment/munkres/munkres.cpp 
    )

target_link_libraries(${TARGET_LIBRARY} glib-2.0 gstreamer-1.0 gobject-2.0 gstbase-1.0 gmodule-2.0 glog)

target_link_libraries(${TARGET_LIBRARY} mxpidatatype plugintoolkit mxbase mindxsdk_protobuf)

