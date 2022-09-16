cmake_minimum_required(VERSION 3.10)
project(mxpi_featurematch)

set(CMAKE_CXX_STANDARD 11)

set(PLUGIN_NAME "mxpi_featurematch")
set(TARGET_LIBRARY ${PLUGIN_NAME})

add_compile_options(-fPIC -fstack-protector-all -g -Wl,-z,relro,-z,now,-z -pie -Wall)
add_compile_options(-std=c++11 -Wno-deprecated-declarations)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)
add_definitions(-DENABLE_DVPP_INTERFACE)


set(MX_SDK_HOME "$ENV{MX_SDK_HOME}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${MX_SDK_HOME}/lib/plugins/)

include_directories(${MX_SDK_HOME}/include)
include_directories(${MX_SDK_HOME}/opensource/include)
include_directories(${MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories(${MX_SDK_HOME}/opensource/include/opencv4)
include_directories(${MX_SDK_HOME}/opensource/include/glib-2.0)
include_directories(${MX_SDK_HOME}/opensource/lib/glib-2.0/include)


link_directories(${MX_SDK_HOME}/lib)
link_directories(${MX_SDK_HOME}/opensource/lib)

file(GLOB PLUGIN_SRC ./*.cpp)
message(${PLUGIN_SRC})

add_library(${TARGET_LIBRARY} SHARED ${PLUGIN_SRC})
target_link_libraries(${TARGET_LIBRARY}
        mxpidatatype
        plugintoolkit
        mxbase
        streammanager
        mindxsdk_protobuf
        glib-2.0
        gstreamer-1.0
        gobject-2.0
        gstbase-1.0
        gmodule-2.0
        )
