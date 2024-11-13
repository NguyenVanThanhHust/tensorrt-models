cmake_minimum_required(VERSION 3.10)  # change the version, if asked by compiler
project(yolov3)

set(CMAKE_CXX_STANDARD 17)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE Debug) # Change this later for release version

find_package(CUDA REQUIRED)

# include and link dirs of tensorrt, you need adapt them if yours are different
include_directories(/opt/TensorRT-8.6.1.6/include)
link_directories(/opt/TensorRT-8.6.1.6/lib)
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# include and link dirs of cuda for inference
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(yololayer nvinfer cudart ${OpenCV_LIBS})

add_executable(yolov3 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/yolov3.cpp)
target_link_libraries(yolov3 nvinfer)
target_link_libraries(yolov3 cudart)
target_link_libraries(yolov3 yololayer)
target_link_libraries(yolov3 ${OpenCV_LIBS})

add_definitions(-O2 -pthread)