cmake_minimum_required(VERSION 3.10)  # change the version, if asked by compiler
project(mlp)

set(CMAKE_CXX_STANDARD 14)

# include and link dirs of tensorrt, you need adapt them if yours are different
include_directories(/opt/TensorRT-8.6.1.6/include)
link_directories(/opt/TensorRT-8.6.1.6/lib)

# include and link dirs of cuda for inference
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# Create link for executable files
add_executable(build_engine build_engine.cpp)

# linking with nvinfer libraries
target_link_libraries(build_engine nvinfer)

# linking with cuda for inference
target_link_libraries(build_engine cudart)

# Create link for executable files
add_executable(infer infer.cpp)

# linking with nvinfer libraries
target_link_libraries(infer nvinfer)

# linking with cuda for inference
target_link_libraries(infer cudart)

add_definitions(-O2 -pthread)