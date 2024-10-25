# Minimum CMake version
cmake_minimum_required(VERSION 3.10)
# Required packages and flags
## CUDA
set(CMAKE_CUDA_COMPILER "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4")
enable_language(CUDA)  # 启用CUDA支持
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message("CUDA FOUND, Version: " ${CUDA_VERSION_STRING})
endif ()

SET(CMAKE_CXX_COMPILER pgc++)

## CUDA
list(APPEND CUDA_NVCC_FLAGS "-fmad=false")

cuda_add_executable(cuda
        cuda.cu
        ../matrix.cpp ../matrix.hpp)
target_link_libraries(cuda cudart )
set_target_properties(cuda PROPERTIES LINK_FLAGS "-Wl,-rpath,/opt/nvidia/hpc_sdk/Linux_x86_64/21.7/cuda/11.4/lib64")
