#!/bin/bash

# PatchCore C++ Build Script
# This script builds the PatchCore C++ implementation

set -e

echo "Building PatchCore C++ implementation..."

# Check if build directory exists
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

# Configure CMake
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/path/to/libtorch \
    -DOpenCV_DIR=/path/to/opencv/build \
    -DFAISS_ROOT=/path/to/faiss

# Build the project
echo "Building project..."
make -j$(nproc)

echo "Build completed successfully!"
echo "Executable: ./build/patchcore_detector"

# Create a simple test
echo "Running basic test..."
./patchcore_detector --help

echo "PatchCore C++ implementation is ready!"



