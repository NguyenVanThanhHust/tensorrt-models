cmake_minimum_required(VERSION 3.1)

project(yolov3)

add_definitions(-std=c++14)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
include_directories(/workspace/TensorRT-8.4.3.1/include)
link_directories(/workspace/TensorRT-8.4.3.1/lib)

# opencv
include_directories(/usr/local/include/opencv4/)
link_directories(/usr/local/lib/)


# find_package(OpenCV)
# include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

#cuda_add_library(leaky ${PROJECT_SOURCE_DIR}/leaky.cu)
cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(yololayer nvinfer cudart ${OpenCV_LIBS})

add_executable(yolov3 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/yolov3.cpp)
target_link_libraries(yolov3 nvinfer)
target_link_libraries(yolov3 cudart)
target_link_libraries(yolov3 yololayer)
# target_link_libraries(yolov3 ${OpenCV_LIBS})
target_link_libraries(yolov3 /usr/local/lib/libopencv_core.so)
target_link_libraries(yolov3 /usr/local/lib/libopencv_imgproc.so)
target_link_libraries(yolov3 /usr/local/lib/libopencv_imgcodecs.so)
target_link_libraries(yolov3 /usr/local/lib/libopencv_highgui.so)
target_link_libraries(yolov3 /usr/local/lib/libopencv_videoio.so)
target_link_libraries(yolov3 /usr/local/lib/libopencv_video.so)
target_link_libraries(yolov3 /usr/local/lib/libopencv_dnn.so)

add_definitions(-O2 -pthread)