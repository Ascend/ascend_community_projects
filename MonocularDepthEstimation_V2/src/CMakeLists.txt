# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)
# project information
project(MonocularDepthEstimation)

# Compile options
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0 -Dgoogle=mindxsdk_private)
add_compile_options(-std=c++11 -fPIC -fstack-protector-all -Wall -D_FORTIFY_SOURCE=2 -O2)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../../")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,relro,-z,now,-z,noexecstack -s -pie")
set(CMAKE_SKIP_RPATH TRUE)

# Header path
include_directories(
    ${MX_SDK_HOME}/include/
    ${MX_SDK_HOME}/opensource/include/
    ${MX_SDK_HOME}/opensource/include/opencv4/
    ./
)

# add host lib path
link_directories(
    ${MX_SDK_HOME}/lib/
    ${MX_SDK_HOME}/opensource/lib/
    ${MX_SDK_HOME}/opensource/lib64/
)

aux_source_directory(. sourceList)

add_executable(main ${sourceList})

target_link_libraries(main mxbase opencv_world boost_filesystem glog)

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
