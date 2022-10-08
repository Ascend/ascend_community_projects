cmake_minimum_required(VERSION 3.5.1)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)
set(PLUGIN_NAME "animeganpostprocessor")
set(TARGET_LIBRARY ${PLUGIN_NAME})
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib/plugins)
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")
set(CMAKE_SKIP_RPATH TRUE)
set(MX_SDK_HOME $ENV{MX_SDK_HOME})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${MX_SDK_HOME}/include)
include_directories(${MX_SDK_HOME}/opensource/include)
include_directories(${MX_SDK_HOME}/opensource/include/opencv4)
include_directories(${MX_SDK_HOME}/opensource/include/gstreamer-1.0)
include_directories(${MX_SDK_HOME}/opensource/include/glib-2.0)
include_directories(${MX_SDK_HOME}/opensource/lib/glib-2.0/include)

link_directories(${MX_SDK_HOME}/lib)

add_compile_options(-std=c++11 -fPIC -fstack-protector-all -Wno-deprecated-declarations -D_FORTIFY_SOURCE=2 -O2)
add_compile_options("-DPLUGIN_NAME=${PLUGIN_NAME}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -pie")
add_library(${TARGET_LIBRARY} SHARED AnimeGANPostProcessor.cpp)

target_link_libraries(${TARGET_LIBRARY} plugintoolkit mxbase)
