project(quadperf)


cmake_minimum_required(VERSION 2.8.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/3rdparty/visionaray/cmake"
    "${PROJECT_SOURCE_DIR}/3rdparty/visionaray/cmake/modules"
)


include(VisionarayAddTarget)
include(VisionarayUseFramework)
include(VisionarayUsePackage)

set(VSNRAY_ENABLE_VIEWER FALSE CACHE BOOL "Visionaray viewer" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
add_subdirectory(3rdparty/visionaray)


find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)
if (VSNRAY_ENABLE_CUDA)
find_package(CUDA)
endif()
if (VSNRAY_ENABLE_CUDA)
visionaray_use_package(CUDA)
endif()


set(VISIONARAY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/visionaray)
set(VISIONARAY_INCLUDE_DIR ${VISIONARAY_DIR}/include)
set(VISIONARAY_SOURCE_DIR ${VISIONARAY_DIR}/src)

set(CMD_LINE_DIR ${VISIONARAY_DIR}//src/3rdparty/CmdLine)
set(CMD_LINE_INCLUDE_DIR ${CMD_LINE_DIR}/include)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/3rdparty/visionaray/config)
include_directories(${VISIONARAY_INCLUDE_DIR})
include_directories(${VISIONARAY_SOURCE_DIR})
include_directories(${CMD_LINE_INCLUDE_DIR})


### QUADEX ###

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    visionaray_cuda_compile(EX_QUAD_CUDA_SOURCES
        main.cu
    )
else()
    set(EX_QUAD_SOURCES
        main.cpp
    )
endif()


visionaray_link_libraries(visionaray)
visionaray_link_libraries(visionaray_common)

visionaray_add_executable(quadbench
    ${EX_QUAD_SOURCES}
    ${EX_QUAD_CUDA_SOURCES}
)
