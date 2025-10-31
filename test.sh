#!/bin/bash

# PatchCore C++ Test Script
# This script tests the PatchCore C++ implementation

set -e

echo "Testing PatchCore C++ implementation..."

# Check if executable exists
if [ ! -f "build/patchcore_detector" ]; then
    echo "Error: patchcore_detector not found. Please build the project first."
    echo "Run: ./build.sh"
    exit 1
fi

# Create test directories
echo "Creating test directories..."
mkdir -p test_data/normal_images
mkdir -p test_data/test_images
mkdir -p test_output

# Create sample model if it doesn't exist
if [ ! -f "models/patchcore_extractor.pt" ]; then
    echo "Creating sample model..."
    python3 convert_model.py --create-sample --output models/patchcore_extractor.pt
fi

# Create sample images (if OpenCV is available)
echo "Creating sample test images..."
python3 -c "
import cv2
import numpy as np
import os

# Create sample normal images
for i in range(5):
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(f'test_data/normal_images/normal_{i:03d}.jpg', img)

# Create sample test images
for i in range(3):
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(f'test_data/test_images/test_{i:03d}.jpg', img)

print('Sample images created successfully!')
"

# Test basic functionality
echo "Testing basic functionality..."
./build/patchcore_detector \
    models/patchcore_extractor.pt \
    test_data/normal_images \
    test_data/test_images \
    --metric cosine \
    --core-set-size 1000 \
    --k-neighbors 1 \
    --aggregation mean \
    --output-dir test_output \
    --cpu

# Check if output files were created
if [ -f "test_output/anomaly_scores.txt" ]; then
    echo "✓ Anomaly scores file created"
    cat test_output/anomaly_scores.txt
else
    echo "✗ Anomaly scores file not found"
fi

if [ -d "test_output" ] && [ "$(ls -A test_output)" ]; then
    echo "✓ Output directory contains files"
    ls -la test_output/
else
    echo "✗ Output directory is empty"
fi

# Test with different metrics
echo "Testing with L2 distance metric..."
./build/patchcore_detector \
    models/patchcore_extractor.pt \
    test_data/normal_images \
    test_data/test_images \
    --metric l2 \
    --core-set-size 1000 \
    --k-neighbors 3 \
    --aggregation max \
    --output-dir test_output_l2 \
    --cpu

# Test with different aggregation methods
echo "Testing with different aggregation methods..."
for agg in mean max median weighted_mean; do
    echo "Testing aggregation: $agg"
    ./build/patchcore_detector \
        models/patchcore_extractor.pt \
        test_data/normal_images \
        test_data/test_images \
        --metric cosine \
        --core-set-size 1000 \
        --k-neighbors 2 \
        --aggregation $agg \
        --output-dir test_output_$agg \
        --cpu
done

echo "All tests completed successfully!"
echo "Test results:"
echo "- Basic test: test_output/"
echo "- L2 metric test: test_output_l2/"
echo "- Aggregation tests: test_output_*"

# Cleanup (optional)
read -p "Do you want to clean up test files? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleaning up test files..."
    rm -rf test_data test_output*
    echo "Cleanup completed!"
fi



