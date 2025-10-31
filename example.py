#!/usr/bin/env python3
"""
Example script demonstrating PatchCore C++ implementation usage
"""

import subprocess
import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path


def create_sample_data(output_dir, num_normal=10, num_test=5):
    """
    Create sample data for testing
    """
    print(f"Creating sample data in {output_dir}...")
    
    # Create directories
    normal_dir = Path(output_dir) / "normal_images"
    test_dir = Path(output_dir) / "test_images"
    normal_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create normal images (random patterns)
    for i in range(num_normal):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(normal_dir / f"normal_{i:03d}.jpg"), img)
    
    # Create test images (some with anomalies)
    for i in range(num_test):
        if i < 2:  # First 2 images are normal
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        else:  # Rest have anomalies
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            # Add anomaly (bright spot)
            cv2.circle(img, (128, 128), 20, (255, 255, 255), -1)
        
        cv2.imwrite(str(test_dir / f"test_{i:03d}.jpg"), img)
    
    print(f"Created {num_normal} normal images and {num_test} test images")
    return str(normal_dir), str(test_dir)


def run_patchcore_detector(model_path, normal_dir, test_dir, output_dir, 
                          metric="cosine", core_set_size=10000, k_neighbors=1, 
                          aggregation="mean", use_gpu=True):
    """
    Run PatchCore detector
    """
    print(f"Running PatchCore detector...")
    print(f"Model: {model_path}")
    print(f"Normal images: {normal_dir}")
    print(f"Test images: {test_dir}")
    print(f"Output: {output_dir}")
    print(f"Metric: {metric}")
    print(f"Core set size: {core_set_size}")
    print(f"K neighbors: {k_neighbors}")
    print(f"Aggregation: {aggregation}")
    print(f"GPU: {use_gpu}")
    
    # Build command
    cmd = [
        "./build/patchcore_detector",
        model_path,
        normal_dir,
        test_dir,
        "--metric", metric,
        "--core-set-size", str(core_set_size),
        "--k-neighbors", str(k_neighbors),
        "--aggregation", aggregation,
        "--output-dir", output_dir
    ]
    
    if use_gpu:
        cmd.append("--gpu")
    else:
        cmd.append("--cpu")
    
    # Run command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ PatchCore detector completed successfully")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ PatchCore detector failed with exit code {e.returncode}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def analyze_results(output_dir):
    """
    Analyze PatchCore results
    """
    print(f"Analyzing results in {output_dir}...")
    
    # Read anomaly scores
    scores_file = Path(output_dir) / "anomaly_scores.txt"
    if scores_file.exists():
        print("Anomaly Scores:")
        print("==============")
        with open(scores_file, 'r') as f:
            for line in f:
                print(line.strip())
    else:
        print("✗ Anomaly scores file not found")
    
    # List output files
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"\nOutput files in {output_dir}:")
        for file in output_path.iterdir():
            if file.is_file():
                print(f"  {file.name}")
    else:
        print(f"✗ Output directory {output_dir} not found")


def compare_metrics(model_path, normal_dir, test_dir, output_base_dir):
    """
    Compare different distance metrics
    """
    print("Comparing distance metrics...")
    
    metrics = ["cosine", "l2"]
    results = {}
    
    for metric in metrics:
        output_dir = f"{output_base_dir}_metric_{metric}"
        print(f"\nTesting metric: {metric}")
        
        success = run_patchcore_detector(
            model_path, normal_dir, test_dir, output_dir,
            metric=metric, core_set_size=5000, k_neighbors=3,
            aggregation="mean", use_gpu=False
        )
        
        if success:
            results[metric] = output_dir
            analyze_results(output_dir)
    
    return results


def compare_aggregations(model_path, normal_dir, test_dir, output_base_dir):
    """
    Compare different aggregation methods
    """
    print("Comparing aggregation methods...")
    
    aggregations = ["mean", "max", "median", "weighted_mean"]
    results = {}
    
    for agg in aggregations:
        output_dir = f"{output_base_dir}_agg_{agg}"
        print(f"\nTesting aggregation: {agg}")
        
        success = run_patchcore_detector(
            model_path, normal_dir, test_dir, output_dir,
            metric="cosine", core_set_size=5000, k_neighbors=3,
            aggregation=agg, use_gpu=False
        )
        
        if success:
            results[agg] = output_dir
            analyze_results(output_dir)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='PatchCore C++ Example')
    parser.add_argument('--model', type=str, default='models/patchcore_extractor.pt',
                       help='Path to LibTorch model')
    parser.add_argument('--data-dir', type=str, default='example_data',
                       help='Directory for sample data')
    parser.add_argument('--output-dir', type=str, default='example_output',
                       help='Output directory')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample model')
    parser.add_argument('--create-data', action='store_true',
                       help='Create sample data')
    parser.add_argument('--compare-metrics', action='store_true',
                       help='Compare different distance metrics')
    parser.add_argument('--compare-aggregations', action='store_true',
                       help='Compare different aggregation methods')
    parser.add_argument('--core-set-size', type=int, default=10000,
                       help='Core set size')
    parser.add_argument('--k-neighbors', type=int, default=1,
                       help='Number of neighbors')
    parser.add_argument('--use-gpu', action='store_true',
                       help='Use GPU acceleration')
    
    args = parser.parse_args()
    
    # Create sample model if requested
    if args.create_sample:
        print("Creating sample model...")
        subprocess.run([
            sys.executable, "convert_model.py", 
            "--create-sample", 
            "--output", args.model
        ], check=True)
    
    # Create sample data if requested
    if args.create_data:
        normal_dir, test_dir = create_sample_data(args.data_dir)
    else:
        normal_dir = os.path.join(args.data_dir, "normal_images")
        test_dir = os.path.join(args.data_dir, "test_images")
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model {args.model} not found")
        print("Run with --create-sample to create a sample model")
        return 1
    
    # Check if data directories exist
    if not os.path.exists(normal_dir) or not os.path.exists(test_dir):
        print(f"Error: Data directories not found")
        print("Run with --create-data to create sample data")
        return 1
    
    # Run basic example
    print("Running basic PatchCore example...")
    success = run_patchcore_detector(
        args.model, normal_dir, test_dir, args.output_dir,
        metric="cosine", core_set_size=args.core_set_size,
        k_neighbors=args.k_neighbors, aggregation="mean",
        use_gpu=args.use_gpu
    )
    
    if success:
        analyze_results(args.output_dir)
    
    # Compare metrics if requested
    if args.compare_metrics:
        compare_metrics(args.model, normal_dir, test_dir, args.output_dir)
    
    # Compare aggregations if requested
    if args.compare_aggregations:
        compare_aggregations(args.model, normal_dir, test_dir, args.output_dir)
    
    print("\nExample completed successfully!")
    return 0


if __name__ == '__main__':
    exit(main())

