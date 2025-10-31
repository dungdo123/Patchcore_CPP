#include "patchcore.h"
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

// Helper function to create visualization with overlay
cv::Mat createVisualization(const cv::Mat& original_image, const cv::Mat& anomaly_map, float alpha, float anomaly_score) {
    // Resize anomaly map to match original image size using bilinear interpolation
    // Bilinear is more appropriate for upsampling small feature maps
    cv::Mat resized_anomaly_map;
    cv::resize(anomaly_map, resized_anomaly_map, original_image.size(), 0, 0, cv::INTER_LINEAR);
    
    cv::Mat heatmap;
    
    // Check if anomaly_map is already a colormapped image (3-channel BGR)
    if (resized_anomaly_map.channels() == 3 && resized_anomaly_map.type() == CV_8UC3) {
        // Already colormapped, use directly
        heatmap = resized_anomaly_map;
        heatmap.convertTo(heatmap, CV_32F, 1.0 / 255.0);
    } else {
        // Convert anomaly map to single channel for colormap
        cv::Mat normalized_gray;
        if (resized_anomaly_map.channels() == 3) {
            cv::cvtColor(resized_anomaly_map, normalized_gray, cv::COLOR_BGR2GRAY);
        } else {
            normalized_gray = resized_anomaly_map;
        }
        
        // Apply Gaussian blur to smooth the anomaly map and reduce grid artifacts
        // Use a smaller kernel to preserve more detail (3x3 instead of 5x5)
        cv::Mat blurred;
        cv::GaussianBlur(normalized_gray, blurred, cv::Size(3, 3), 0.5);
        
        // Normalize to [0, 1] using percentile-based normalization for better handling of outliers
        cv::Mat normalized;
        if (blurred.type() == CV_32F || blurred.type() == CV_64F) {
            // Use 99th percentile instead of max for more robust normalization
            // Higher percentile better highlights true anomalies vs normal variations
            cv::Mat flattened;
            blurred.reshape(1, 1).copyTo(flattened);
            flattened.convertTo(flattened, CV_32F);
            
            std::vector<float> values;
            flattened.reshape(1, flattened.total()).copyTo(values);
            std::sort(values.begin(), values.end());
            
            double min_val = values[0];
            double max_val = values[(std::min)(static_cast<size_t>(values.size() * 0.99), values.size() - 1)];
            
            // Normalize to [0, 1] using percentile-based max
            blurred.convertTo(normalized, CV_32F);
            double range = (std::max)(max_val - min_val, 1e-6);
            normalized = (normalized - min_val) / range;
            cv::threshold(normalized, normalized, 0.0, 0.0, cv::THRESH_TOZERO);  // Clamp negatives to 0
            cv::threshold(normalized, normalized, 1.0, 1.0, cv::THRESH_TRUNC);    // Clamp >1 to 1
        } else {
            // Convert to float first
            cv::Mat blurred_float;
            blurred.convertTo(blurred_float, CV_32F);
            
            // Use percentile-based normalization
            cv::Mat flattened;
            blurred_float.reshape(1, 1).copyTo(flattened);
            
            std::vector<float> values;
            flattened.reshape(1, flattened.total()).copyTo(values);
            std::sort(values.begin(), values.end());
            
            double min_val = values[0];
            double max_val = values[(std::min)(static_cast<size_t>(values.size() * 0.99), values.size() - 1)];
            
            double range = (std::max)(max_val - min_val, 1e-6);
            normalized = (blurred_float - min_val) / range;
            cv::threshold(normalized, normalized, 0.0, 0.0, cv::THRESH_TOZERO);  // Clamp negatives to 0
            cv::threshold(normalized, normalized, 1.0, 1.0, cv::THRESH_TRUNC);    // Clamp >1 to 1
        }
        
        // Convert to uint8 for colormap (must be single channel)
        cv::Mat gray_uint8;
        normalized.convertTo(gray_uint8, CV_8U, 255.0);
        
        // Apply colormap
        cv::applyColorMap(gray_uint8, heatmap, cv::COLORMAP_JET);
        heatmap.convertTo(heatmap, CV_32F, 1.0 / 255.0);
    }
    
    // Convert original image to float
    cv::Mat original_float;
    original_image.convertTo(original_float, CV_32F, 1.0 / 255.0);
    
    // Blend original image with heatmap
    cv::Mat blended;
    cv::addWeighted(original_float, 1.0 - alpha, heatmap, alpha, 0.0, blended);
    
    // Convert back to uint8
    cv::Mat output;
    blended.convertTo(output, CV_8U, 255.0);
    
    // Add text overlay with anomaly score
    std::stringstream ss;
    ss << "Anomaly Score: " << std::fixed << std::setprecision(4) << anomaly_score;
    std::string score_text = ss.str();
    
    // Determine text color based on score (red for high, green for low)
    cv::Scalar text_color;
    if (anomaly_score > 0.5) {
        text_color = cv::Scalar(0, 0, 255); // Red for high anomaly
    } else {
        text_color = cv::Scalar(0, 255, 0); // Green for low anomaly
    }
    
    // Calculate text size and position
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.8;
    int thickness = 2;
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(score_text, font_face, font_scale, thickness, &baseline);
    
    // Position text at top-left corner with padding
    cv::Point text_pos(10, text_size.height + 10);
    
    // Draw background rectangle for text
    cv::rectangle(output, 
                  cv::Point(text_pos.x - 5, text_pos.y - text_size.height - 5),
                  cv::Point(text_pos.x + text_size.width + 5, text_pos.y + baseline + 5),
                  cv::Scalar(0, 0, 0), -1); // Black background
    
    // Draw text
    cv::putText(output, score_text, text_pos, font_face, font_scale, text_color, thickness);
    
    return output;
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    // Try to load CUDA library if available (optional)
    HMODULE loadLib = LoadLibraryA("torch_cuda.dll");
    if (loadLib == NULL) {
        std::cout << "CUDA library not found, using CPU only" << std::endl;
    } else {
        std::cout << "CUDA library loaded successfully" << std::endl;
    }
#endif

    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <model_path> <memory_bank_path> <input_image> <output_image> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --metric cosine|l2     Distance metric (must match training, default: l2)" << std::endl;
        std::cout << "  --k-neighbors N       Number of neighbors (default: 1)" << std::endl;
        std::cout << "  --aggregation method  Aggregation method: mean|max|median|weighted_mean (default: mean)" << std::endl;
        std::cout << "  --alpha F             Overlay transparency (0.0-1.0, default: 0.5)" << std::endl;
        std::cout << "  --cpu                 Use CPU instead of GPU" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " model.pt memory_bank test_image.jpg output.jpg --alpha 0.6" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string memory_bank_path = argv[2];
    std::string input_image_path = argv[3];
    std::string output_image_path = argv[4];
    
    // Parse command line arguments
    patchcore::DistanceMetric metric = patchcore::DistanceMetric::L2;
    int k_neighbors = 1;
    std::string aggregation = "mean";
    float overlay_alpha = 0.5f;
    bool use_gpu = true;
    
    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--metric" && i + 1 < argc) {
            std::string metric_str = argv[++i];
            if (metric_str == "l2") {
                metric = patchcore::DistanceMetric::L2;
            } else if (metric_str == "cosine") {
                metric = patchcore::DistanceMetric::COSINE;
            } else {
                std::cerr << "Invalid metric: " << metric_str << std::endl;
                return 1;
            }
        } else if (arg == "--k-neighbors" && i + 1 < argc) {
            k_neighbors = std::stoi(argv[++i]);
        } else if (arg == "--aggregation" && i + 1 < argc) {
            aggregation = argv[++i];
        } else if (arg == "--alpha" && i + 1 < argc) {
            overlay_alpha = std::stof(argv[++i]);
            // Clamp to [0, 1] - use (std::min) to prevent macro expansion on Windows
            overlay_alpha = (std::max)(0.0f, (std::min)(1.0f, overlay_alpha));
        } else if (arg == "--cpu") {
            use_gpu = false;
        }
    }
    
    // Create configuration (must match training configuration)
    patchcore::PatchCoreConfig config;
    config.model_path = model_path;
    config.layers_to_extract = {};
    config.patch_size = 3;
    config.target_dim = 384;
    config.core_set_size = 10000; // Not used when loading, but needed for config
    config.metric = metric;
    config.auto_size = true;
    config.coverage_threshold = 0.95;
    config.input_size = 256;
    config.k_neighbors = k_neighbors;
    config.aggregation = aggregation;
    config.use_gpu = use_gpu;
    
    std::cout << "PatchCore Inference Configuration:" << std::endl;
    std::cout << "  Model path: " << config.model_path << std::endl;
    std::cout << "  Memory bank path: " << memory_bank_path << std::endl;
    std::cout << "  Input image: " << input_image_path << std::endl;
    std::cout << "  Output image: " << output_image_path << std::endl;
    std::cout << "  Distance metric: " << (metric == patchcore::DistanceMetric::COSINE ? "cosine" : "l2") << std::endl;
    std::cout << "  K neighbors: " << config.k_neighbors << std::endl;
    std::cout << "  Aggregation: " << config.aggregation << std::endl;
    std::cout << "  Overlay alpha: " << overlay_alpha << std::endl;
    std::cout << "  Device: " << (config.use_gpu ? "GPU" : "CPU") << std::endl;
    
    try {
        // Create detector
        patchcore::PatchCoreDetector detector(config);
        
        // Initialize detector
        std::cout << "\nInitializing detector..." << std::endl;
        if (!detector.initialize()) {
            std::cerr << "Failed to initialize detector" << std::endl;
            return 1;
        }
        
        // Load saved memory bank
        std::cout << "\nLoading memory bank from: " << memory_bank_path << std::endl;
        detector.loadMemoryBank(memory_bank_path);
        
        // Load input image
        std::cout << "\nLoading input image..." << std::endl;
        cv::Mat input_image = cv::imread(input_image_path);
        if (input_image.empty()) {
            std::cerr << "Failed to load image: " << input_image_path << std::endl;
            return 1;
        }
        std::cout << "Input image size: " << input_image.cols << "x" << input_image.rows << std::endl;
        
        // Perform inference
        std::cout << "\nRunning anomaly detection..." << std::endl;
        std::cout << "Calling predictSingle..." << std::endl;
        
        patchcore::PatchCoreDetector::PredictionResult result;
        try {
            result = detector.predictSingle(input_image);
            std::cout << "predictSingle completed successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error in predictSingle: " << e.what() << std::endl;
            return 1;
        }
        
        std::cout << "Prediction completed. Result structure:" << std::endl;
        std::cout << "  Image scores shape: [" << result.image_scores.size(0) << "]" << std::endl;
        std::cout << "  Spatial scores shape: [" << result.spatial_scores.size(0) << ", " 
                  << result.spatial_scores.size(1) << ", " << result.spatial_scores.size(2) << "]" << std::endl;
        std::cout << "  Anomaly maps count: " << result.anomaly_maps.size() << std::endl;
        
        // Get anomaly score
        float anomaly_score = result.image_scores.item<float>();
        std::cout << "Anomaly Score: " << anomaly_score << std::endl;
        
        if (result.anomaly_maps.empty()) {
            std::cerr << "Error: No anomaly maps in result!" << std::endl;
            return 1;
        }
        
        std::cout << "Creating visualization..." << std::endl;
        
        // Convert spatial scores tensor to OpenCV Mat for visualization
        // Important: PyTorch tensors are row-major [height, width]
        // OpenCV Mat expects (rows, cols) which is the same, but we need to ensure correct memory layout
        torch::Tensor spatial_scores_cpu = result.spatial_scores[0].cpu().contiguous();
        int height = static_cast<int>(spatial_scores_cpu.size(0));
        int width = static_cast<int>(spatial_scores_cpu.size(1));
        
        // Clone and ensure contiguous layout
        // PyTorch tensor shape [height, width] should map to OpenCV Mat (rows, cols)
        // However, we need to verify the layout is correct
        torch::Tensor scores_cloned = spatial_scores_cpu.clone().contiguous();
        
        // Debug: print some sample values to verify layout
        std::cout << "  Sample spatial scores (first row): ";
        for (int i = 0; i < (std::min)(width, 5); ++i) {
            std::cout << scores_cloned[0][i].item<float>() << " ";
        }
        std::cout << std::endl;
        std::cout << "  Sample spatial scores (first column): ";
        for (int i = 0; i < (std::min)(height, 5); ++i) {
            std::cout << scores_cloned[i][0].item<float>() << " ";
        }
        std::cout << std::endl;
        
        // Create OpenCV Mat - OpenCV uses (rows, cols) which matches [height, width]
        cv::Mat anomaly_map(height, width, CV_32F, scores_cloned.data_ptr<float>());
        
        // Verify the Mat was created correctly by checking a few pixels
        std::cout << "  Anomaly map sample values (0,0)=" << anomaly_map.at<float>(0, 0) 
                  << ", (0,1)=" << anomaly_map.at<float>(0, 1)
                  << ", (1,0)=" << anomaly_map.at<float>(1, 0) << std::endl;
        
        std::cout << "  Spatial scores converted to Mat: " << anomaly_map.cols << "x" << anomaly_map.rows 
                  << ", channels: " << anomaly_map.channels() 
                  << ", type: " << anomaly_map.type() << std::endl;
        
        // Create visualization with overlay
        cv::Mat output_image;
        try {
            output_image = createVisualization(input_image, anomaly_map, overlay_alpha, anomaly_score);
            std::cout << "  Visualization function completed." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error in createVisualization: " << e.what() << std::endl;
            return 1;
        }
        
        if (output_image.empty()) {
            std::cerr << "Error: Failed to create visualization - output image is empty!" << std::endl;
            return 1;
        }
        
        std::cout << "Visualization created. Output image size: " << output_image.cols << "x" << output_image.rows << std::endl;
        
        // Save output image
        std::cout << "\nSaving output image to: " << output_image_path << std::endl;
        
        // Create output directory if needed
        std::filesystem::path output_dir = std::filesystem::path(output_image_path).parent_path();
        if (!output_dir.empty() && !std::filesystem::exists(output_dir)) {
            std::filesystem::create_directories(output_dir);
        }
        
        bool save_success = cv::imwrite(output_image_path, output_image);
        if (!save_success) {
            std::cerr << "Error: Failed to save output image to: " << output_image_path << std::endl;
            return 1;
        }
        
        // Verify file was created
        if (!std::filesystem::exists(output_image_path)) {
            std::cerr << "Error: Output file was not created: " << output_image_path << std::endl;
            return 1;
        }
        
        // Print summary
        std::cout << "\n========================================" << std::endl;
        std::cout << "Inference Results:" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Anomaly Score: " << anomaly_score << std::endl;
        std::cout << "Output saved to: " << output_image_path << std::endl;
        std::cout << "========================================" << std::endl;
        
        std::cout << "\nInference completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

