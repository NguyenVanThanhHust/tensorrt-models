cmake_minimum_required(VERSION 3.10)  # change the version, if asked by compiler
project(mlp)

set(CMAKE_CXX_STANDARD 14)

# include and link dirs of tensorrt, you need adapt them if yours are different
include_directories(/workspace/TensorRT-8.2.4.2/include)
link_directories(/workspace/TensorRT-8.2.4.2/lib)

# include and link dirs of cuda for inference
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# Create link for executable files
add_executable(mlp mlp.cpp)

# linking with nvinfer libraries
target_link_libraries(mlp nvinfer)

# linking with cuda for inference
target_link_libraries(mlp cudart)

add_definitions(-O2 -pthread)