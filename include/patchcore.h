#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <faiss/IndexFlat.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <string>
#include <map>

namespace patchcore {

enum class DistanceMetric {
    COSINE,
    L2
};

struct PatchCoreConfig {
    // Model configuration
    std::string model_path;
    std::vector<std::string> layers_to_extract = {"layer2", "layer3"};
    int patch_size = 3;
    int target_dim = 384;
    
    // Memory bank configuration
    int core_set_size = 10000;
    DistanceMetric metric = DistanceMetric::L2;
    bool auto_size = true;
    double coverage_threshold = 0.95;
    
    // Image processing
    int input_size = 256;
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    
    // Inference configuration
    int k_neighbors = 1;
    std::string aggregation = "mean";
    bool use_gpu = true;
};

class FeatureExtractor {
public:
    FeatureExtractor(const PatchCoreConfig& config);
    ~FeatureExtractor() = default;
    
    // Load model from file
    bool loadModel(const std::string& model_path);
    
    // Extract features from image
    torch::Tensor extractFeatures(const cv::Mat& image);
    
    // Extract features from batch of images
    torch::Tensor extractFeaturesBatch(const std::vector<cv::Mat>& images);
    
    // Get model info
    struct ModelInfo {
        std::map<std::string, std::vector<int64_t>> parameters;
    };
    ModelInfo getModelInfo() const;
    
private:
    PatchCoreConfig config_;
    std::shared_ptr<torch::jit::script::Module> model_;
    torch::Device device_;
    
    // Preprocessing
    cv::Mat preprocessImage(const cv::Mat& image);
    torch::Tensor imageToTensor(const cv::Mat& image);
    
    // Feature extraction
    torch::Tensor extractLayerFeatures(torch::Tensor input, const std::string& layer_name);
    torch::Tensor upsampleToTargetResolution(torch::Tensor features, int target_size);
    torch::Tensor patchifyFeatures(torch::Tensor features);
    torch::Tensor projectFeatures(torch::Tensor features);
};

class MemoryBank {
public:
    MemoryBank(const PatchCoreConfig& config);
    ~MemoryBank() = default;
    
    // Build memory bank from normal images
    void buildMemoryBank(const std::vector<cv::Mat>& normal_images, 
                        FeatureExtractor& extractor);
    
    // Add features to memory bank
    void addFeatures(const torch::Tensor& features);
    
    // Build core set using greedy sampling
    void buildCoreSet();
    
    // Build core set for a batch of features
    torch::Tensor buildCoreSetForBatch(const torch::Tensor& batch_features);
    
    // Compute anomaly scores
    torch::Tensor computeAnomalyScores(const torch::Tensor& test_features);
    
    // Get memory bank info
    struct MemoryBankInfo {
        int total_features;
        int core_set_size;
        int feature_dim;
        std::string metric;
    };
    MemoryBankInfo getInfo() const;
    
    // Save memory bank to disk
    void saveMemoryBank(const std::string& save_path);
    
    // Load memory bank from disk
    void loadMemoryBank(const std::string& load_path);
    
private:
    PatchCoreConfig config_;
    torch::Tensor all_features_;
    torch::Tensor core_set_;
    std::unique_ptr<faiss::Index> faiss_index_;
    
    // Core set sampling
    torch::Tensor greedyCoreSetSampling(const torch::Tensor& features);
    torch::Tensor approximateGreedyCoreSet(const torch::Tensor& features);
    
    // Distance computation
    torch::Tensor computeDistances(const torch::Tensor& query_features);
    torch::Tensor aggregateDistances(const torch::Tensor& distances);
    
    // FAISS operations
    void buildFAISSIndex();
    void searchFAISS(const torch::Tensor& query_features, 
                    torch::Tensor& distances, 
                    torch::Tensor& indices);
};

class PatchCoreDetector {
public:
    PatchCoreDetector(const PatchCoreConfig& config);
    ~PatchCoreDetector() = default;
    
    // Initialize detector
    bool initialize();
    
    // Train on normal images
    void train(const std::vector<cv::Mat>& normal_images);
    
    // Predict anomaly scores
    struct PredictionResult {
        torch::Tensor image_scores;
        torch::Tensor spatial_scores;
        std::vector<cv::Mat> anomaly_maps;
    };
    PredictionResult predict(const std::vector<cv::Mat>& test_images);
    
    // Predict single image
    PredictionResult predictSingle(const cv::Mat& image);
    
    // Save/load model
    void saveModel(const std::string& path);
    void loadModel(const std::string& path);
    
    // Save/load memory bank
    void saveMemoryBank(const std::string& path);
    void loadMemoryBank(const std::string& path);
    
    // Get detector info
    struct DetectorInfo {
        PatchCoreConfig config;
        FeatureExtractor::ModelInfo model_info;
        MemoryBank::MemoryBankInfo memory_bank_info;
    };
    DetectorInfo getInfo() const;
    
private:
    PatchCoreConfig config_;
    std::unique_ptr<FeatureExtractor> feature_extractor_;
    std::unique_ptr<MemoryBank> memory_bank_;
    bool is_trained_;
    
    // Visualization
    cv::Mat createAnomalyMap(const torch::Tensor& spatial_scores, 
                           const cv::Size& original_size);
    std::vector<cv::Mat> createAnomalyMaps(const torch::Tensor& spatial_scores,
                                          const std::vector<cv::Mat>& original_images);
};

// Utility functions
namespace utils {
    // Image processing
    std::vector<cv::Mat> loadImagesFromDirectory(const std::string& directory_path);
    cv::Mat resizeImage(const cv::Mat& image, int target_size);
    
    // Tensor operations
    torch::Tensor normalizeTensor(const torch::Tensor& tensor, 
                                const std::vector<float>& mean, 
                                const std::vector<float>& std);
    torch::Tensor denormalizeTensor(const torch::Tensor& tensor,
                                  const std::vector<float>& mean,
                                  const std::vector<float>& std);
    
    // Distance metrics
    torch::Tensor cosineDistance(const torch::Tensor& a, const torch::Tensor& b);
    torch::Tensor l2Distance(const torch::Tensor& a, const torch::Tensor& b);
    
    // Aggregation methods
    torch::Tensor meanAggregation(const torch::Tensor& distances);
    torch::Tensor maxAggregation(const torch::Tensor& distances);
    torch::Tensor medianAggregation(const torch::Tensor& distances);
    torch::Tensor weightedMeanAggregation(const torch::Tensor& distances);
}

} // namespace patchcore
