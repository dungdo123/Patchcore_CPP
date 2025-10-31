#include "patchcore.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace patchcore {

PatchCoreDetector::PatchCoreDetector(const PatchCoreConfig& config) 
    : config_(config), is_trained_(false) {
}

bool PatchCoreDetector::initialize() {
    try {
        // Initialize feature extractor
        feature_extractor_ = std::make_unique<FeatureExtractor>(config_);
        
        // Load model
        if (!feature_extractor_->loadModel(config_.model_path)) {
            std::cerr << "Failed to load model from: " << config_.model_path << std::endl;
            return false;
        }
        
        // Initialize memory bank
        memory_bank_ = std::make_unique<MemoryBank>(config_);
        
        std::cout << "PatchCore detector initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing detector: " << e.what() << std::endl;
        return false;
    }
}

void PatchCoreDetector::train(const std::vector<cv::Mat>& normal_images) {
    if (!feature_extractor_ || !memory_bank_) {
        throw std::runtime_error("Detector not initialized");
    }
    
    std::cout << "Training PatchCore detector on " << normal_images.size() << " normal images..." << std::endl;
    
    // Build memory bank
    memory_bank_->buildMemoryBank(normal_images, *feature_extractor_);
    
    is_trained_ = true;
    
    std::cout << "Training completed successfully" << std::endl;
}

PatchCoreDetector::PredictionResult PatchCoreDetector::predict(const std::vector<cv::Mat>& test_images) {
    if (!is_trained_) {
        throw std::runtime_error("Detector not trained");
    }
    
    std::cout << "Predicting anomaly scores for " << test_images.size() << " test images..." << std::endl;
    
    std::vector<torch::Tensor> all_spatial_scores;
    std::vector<torch::Tensor> image_scores;
    
    for (size_t i = 0; i < test_images.size(); ++i) {
        if (i % 10 == 0) {
            std::cout << "Processing image " << i + 1 << "/" << test_images.size() << std::endl;
        }
        
        // Extract features
        torch::Tensor features = feature_extractor_->extractFeatures(test_images[i]);
        
        // Flatten spatial dimensions
        int64_t batch_size = features.size(0);
        int64_t channels = features.size(1);
        int64_t height = features.size(2);
        int64_t width = features.size(3);
        
        torch::Tensor flattened = features.view({batch_size * height * width, channels});
        
        std::cout << "  Flattened features shape: [" << flattened.size(0) << ", " << flattened.size(1) << "]" << std::endl;
        std::cout << "  Computing anomaly scores..." << std::endl;
        
        // Compute anomaly scores
        torch::Tensor scores = memory_bank_->computeAnomalyScores(flattened);
        
        std::cout << "  Anomaly scores computed. Shape: [" << scores.size(0) << "]" << std::endl;
        
        // Reshape back to spatial dimensions
        // Ensure scores are contiguous before reshaping to preserve correct spatial layout
        torch::Tensor scores_contiguous = scores.contiguous();
        torch::Tensor spatial_scores = scores_contiguous.view({height, width});
        
        // Verify spatial layout is correct (debug output)
        std::cout << "  Spatial scores min: " << spatial_scores.min().item<float>() 
                  << ", max: " << spatial_scores.max().item<float>()
                  << ", mean: " << spatial_scores.mean().item<float>() << std::endl;
        
        std::cout << "  Spatial scores shape: [" << height << ", " << width << "]" << std::endl;
        all_spatial_scores.push_back(spatial_scores);
        
        // Compute image-level score (max over spatial dimensions)
        auto max_result = torch::max(spatial_scores);
        torch::Tensor image_score = max_result;
        image_scores.push_back(image_score);
    }
    
    // Stack results
    torch::Tensor stacked_image_scores = torch::stack(image_scores);
    torch::Tensor stacked_spatial_scores = torch::stack(all_spatial_scores);
    
    // Create anomaly maps
    std::vector<cv::Mat> anomaly_maps = createAnomalyMaps(stacked_spatial_scores, test_images);
    
    PredictionResult result;
    result.image_scores = stacked_image_scores;
    result.spatial_scores = stacked_spatial_scores;
    result.anomaly_maps = anomaly_maps;
    
    return result;
}

PatchCoreDetector::PredictionResult PatchCoreDetector::predictSingle(const cv::Mat& image) {
    return predict({image});
}

void PatchCoreDetector::saveModel(const std::string& path) {
    if (!is_trained_) {
        throw std::runtime_error("Detector not trained");
    }
    
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    
    // Save configuration
    // Note: This is a simplified version. In practice, you'd want to serialize
    // the configuration and memory bank data properly
    
    std::cout << "Model saved to: " << path << std::endl;
}

void PatchCoreDetector::loadModel(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }
    
    // Load configuration and memory bank
    // Note: This is a simplified version. In practice, you'd want to deserialize
    // the configuration and memory bank data properly
    
    is_trained_ = true;
    std::cout << "Model loaded from: " << path << std::endl;
}

cv::Mat PatchCoreDetector::createAnomalyMap(const torch::Tensor& spatial_scores, 
                                          const cv::Size& original_size) {
    // Convert tensor to OpenCV Mat
    // Ensure contiguous memory layout for proper OpenCV Mat creation
    torch::Tensor scores_cpu = spatial_scores.cpu().contiguous();
    
    // Clone to ensure we have our own memory (OpenCV Mat doesn't take ownership)
    torch::Tensor scores_cloned = scores_cpu.clone();
    
    // Create OpenCV Mat from tensor data
    // PyTorch: [height, width] -> OpenCV: (rows=height, cols=width)
    int rows = static_cast<int>(scores_cloned.size(0));
    int cols = static_cast<int>(scores_cloned.size(1));
    cv::Mat anomaly_map(rows, cols, CV_32F, scores_cloned.data_ptr<float>());
    
    // Clone the Mat to ensure OpenCV has its own copy (since tensor memory might be freed)
    anomaly_map = anomaly_map.clone();
    
    // Apply light Gaussian blur to smooth the anomaly map (reduced to minimize artifacts)
    cv::Mat smoothed;
    cv::GaussianBlur(anomaly_map, smoothed, cv::Size(3, 3), 0.5);
    
    // Use percentile-based normalization with a higher percentile (99th) to better highlight anomalies
    // Lower percentiles (like 95th) can make normal variations appear as anomalies
    cv::Mat flattened;
    smoothed.reshape(1, 1).copyTo(flattened);
    flattened.convertTo(flattened, CV_32F);
    
    std::vector<float> values;
    flattened.reshape(1, flattened.total()).copyTo(values);
    std::sort(values.begin(), values.end());
    
    double min_val = values[0];
    // Use 99th percentile to better highlight true anomalies
    double max_val = values[(std::min)(static_cast<size_t>(values.size() * 0.99), values.size() - 1)];
    
    std::cout << "  Anomaly map stats: min=" << min_val << ", max(raw)=" << values.back() 
              << ", 99th_percentile=" << max_val << std::endl;
    
    // Normalize to [0, 1] using percentile-based max
    cv::Mat normalized;
    smoothed.convertTo(normalized, CV_32F);
    double range = (std::max)(max_val - min_val, 1e-6);
    normalized = (normalized - min_val) / range;
    cv::threshold(normalized, normalized, 0.0, 0.0, cv::THRESH_TOZERO);  // Clamp negatives to 0
    cv::threshold(normalized, normalized, 1.0, 1.0, cv::THRESH_TRUNC);    // Clamp >1 to 1
    
    // Resize to original image size using bilinear interpolation
    // Bilinear is more appropriate for upsampling small feature maps (32x32 -> 900x900)
    // Cubic can create artifacts when upsampling by large factors
    cv::Mat resized;
    cv::resize(normalized, resized, original_size, 0, 0, cv::INTER_LINEAR);
    
    // Convert to uint8 for colormap (must be single channel)
    cv::Mat resized_uint8;
    resized.convertTo(resized_uint8, CV_8U, 255.0);
    
    // Convert to heatmap
    cv::Mat heatmap;
    cv::applyColorMap(resized_uint8, heatmap, cv::COLORMAP_JET);
    
    return heatmap;
}

std::vector<cv::Mat> PatchCoreDetector::createAnomalyMaps(const torch::Tensor& spatial_scores,
                                                        const std::vector<cv::Mat>& original_images) {
    std::vector<cv::Mat> anomaly_maps;
    
    for (int i = 0; i < spatial_scores.size(0); ++i) {
        torch::Tensor single_scores = spatial_scores[i];
        cv::Size original_size = original_images[i].size();
        
        cv::Mat anomaly_map = createAnomalyMap(single_scores, original_size);
        anomaly_maps.push_back(anomaly_map);
    }
    
    return anomaly_maps;
}

PatchCoreDetector::DetectorInfo PatchCoreDetector::getInfo() const {
    DetectorInfo info;
    info.config = config_;
    
    if (feature_extractor_) {
        info.model_info = feature_extractor_->getModelInfo();
    }
    
    if (memory_bank_) {
        info.memory_bank_info = memory_bank_->getInfo();
    }
    
    return info;
}

void PatchCoreDetector::saveMemoryBank(const std::string& path) {
    if (!memory_bank_) {
        throw std::runtime_error("Memory bank not initialized");
    }
    
    if (!is_trained_) {
        throw std::runtime_error("Cannot save: detector not trained");
    }
    
    memory_bank_->saveMemoryBank(path);
}

void PatchCoreDetector::loadMemoryBank(const std::string& path) {
    if (!memory_bank_) {
        throw std::runtime_error("Memory bank not initialized");
    }
    
    memory_bank_->loadMemoryBank(path);
    is_trained_ = true;
    
    std::cout << "Memory bank loaded. Detector is ready for inference." << std::endl;
}

} // namespace patchcore

