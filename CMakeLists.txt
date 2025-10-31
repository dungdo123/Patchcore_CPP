cmake_minimum_required(VERSION 3.12)
project(PatchCore)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set C++ standard for LibTorch compatibility
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()

# Configure paths
set(TORCH_ROOT "D:/PROJECTS/0.Installation/libtorch")
set(OpenCV_ROOT "D:/PROJECTS/0.Installation/opencv")
set(FAISS_ROOT "D:/PROJECTS/0.Installation/faiss")

# Include directories
include_directories(
    include
    "${TORCH_ROOT}/include"
    "${TORCH_ROOT}/include/torch/csrc/api/include"
    "${OpenCV_ROOT}/build/include"
    "${FAISS_ROOT}/include"
)

# Library directories
link_directories(
    "${TORCH_ROOT}/lib"
    "${OpenCV_ROOT}/build/x64/vc16/lib"
    "${FAISS_ROOT}/lib"
    "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/lib"
    "C:/Program Files (x86)/Intel/oneAPI/2025.1/lib"
)

# Add executables
add_executable(patchcore_detector
    src/main.cpp
    src/feature_extractor.cpp
    src/memory_bank.cpp
    src/patchcore_detector.cpp
    src/utils.cpp
)

add_executable(patchcore_inference
    src/inference.cpp
    src/feature_extractor.cpp
    src/memory_bank.cpp
    src/patchcore_detector.cpp
    src/utils.cpp
)


# Link libraries
target_link_libraries(patchcore_detector 
    "${TORCH_ROOT}/lib/torch.lib"
    "${TORCH_ROOT}/lib/torch_cpu.lib"
    "${TORCH_ROOT}/lib/c10.lib"
    "${OpenCV_ROOT}/build/x64/vc16/lib/opencv_world4100.lib"
    "${FAISS_ROOT}/lib/faiss.lib"
    libircmt.lib
    mkl_intel_lp64_dll.lib
    mkl_intel_thread_dll.lib
    mkl_core_dll.lib
)

target_link_libraries(patchcore_inference 
    "${TORCH_ROOT}/lib/torch.lib"
    "${TORCH_ROOT}/lib/torch_cpu.lib"
    "${TORCH_ROOT}/lib/c10.lib"
    "${OpenCV_ROOT}/build/x64/vc16/lib/opencv_world4100.lib"
    "${FAISS_ROOT}/lib/faiss.lib"
    libircmt.lib
    mkl_intel_lp64_dll.lib
    mkl_intel_thread_dll.lib
    mkl_core_dll.lib
)



# Set properties
set_target_properties(patchcore_detector PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(patchcore_inference PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Print configuration info
message(STATUS "Torch root: ${TORCH_ROOT}")
message(STATUS "OpenCV root: ${OpenCV_ROOT}")
message(STATUS "FAISS root: ${FAISS_ROOT}")

# Installation
install(TARGETS patchcore_detector patchcore_inference
    RUNTIME DESTINATION bin
)