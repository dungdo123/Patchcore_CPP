#include "patchcore.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <algorithm>

namespace patchcore {
namespace utils {

std::vector<cv::Mat> loadImagesFromDirectory(const std::string& directory_path) {
    std::vector<cv::Mat> images;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(directory_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                
                // Check if it's an image file
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                
                if (extension == ".jpg" || extension == ".jpeg" || 
                    extension == ".png" || extension == ".bmp" || 
                    extension == ".tiff" || extension == ".tif") {
                    
                    cv::Mat image = cv::imread(file_path);
                    if (!image.empty()) {
                        images.push_back(image);
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading images from directory: " << e.what() << std::endl;
    }
    
    return images;
}

cv::Mat resizeImage(const cv::Mat& image, int target_size) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(target_size, target_size));
    return resized;
}

torch::Tensor normalizeTensor(const torch::Tensor& tensor, 
                            const std::vector<float>& mean, 
                            const std::vector<float>& std) {
    torch::Tensor normalized = tensor.clone();
    
    for (int i = 0; i < mean.size(); ++i) {
        normalized[0][i] = (normalized[0][i] - mean[i]) / std[i];
    }
    
    return normalized;
}

torch::Tensor denormalizeTensor(const torch::Tensor& tensor,
                              const std::vector<float>& mean,
                              const std::vector<float>& std) {
    torch::Tensor denormalized = tensor.clone();
    
    for (int i = 0; i < mean.size(); ++i) {
        denormalized[0][i] = denormalized[0][i] * std[i] + mean[i];
    }
    
    return denormalized;
}

torch::Tensor cosineDistance(const torch::Tensor& a, const torch::Tensor& b) {
    // Normalize tensors
    torch::Tensor a_norm = torch::nn::functional::normalize(a, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    torch::Tensor b_norm = torch::nn::functional::normalize(b, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    
    // Compute cosine similarity
    torch::Tensor similarity = torch::mm(a_norm, b_norm.t());
    
    // Convert to distance
    return 1.0 - similarity;
}

torch::Tensor l2Distance(const torch::Tensor& a, const torch::Tensor& b) {
    // Compute squared L2 distance
    torch::Tensor a_expanded = a.unsqueeze(1);  // [N, 1, D]
    torch::Tensor b_expanded = b.unsqueeze(0);  // [1, M, D]
    
    torch::Tensor diff = a_expanded - b_expanded;  // [N, M, D]
    torch::Tensor squared_dist = torch::sum(diff * diff, 2);  // [N, M]
    
    // Return L2 distance (not squared)
    return torch::sqrt(squared_dist + 1e-8);
}

torch::Tensor meanAggregation(const torch::Tensor& distances) {
    return torch::mean(distances, 1);
}

torch::Tensor maxAggregation(const torch::Tensor& distances) {
    auto max_result = torch::max(distances, 1);
    return std::get<0>(max_result);
}

torch::Tensor medianAggregation(const torch::Tensor& distances) {
    // Sort distances and take median
    auto sort_result = torch::sort(distances, 1);
    torch::Tensor sorted_distances = std::get<0>(sort_result);
    int64_t k = sorted_distances.size(1);
    
    if (k % 2 == 1) {
        // Odd number of neighbors
        return sorted_distances.index_select(1, torch::tensor({k / 2}).to(sorted_distances.device()));
    } else {
        // Even number of neighbors - take average of middle two
        torch::Tensor median = (sorted_distances.index_select(1, torch::tensor({k / 2 - 1}).to(sorted_distances.device())) +
                               sorted_distances.index_select(1, torch::tensor({k / 2}).to(sorted_distances.device()))) / 2.0;
        return median;
    }
}

torch::Tensor weightedMeanAggregation(const torch::Tensor& distances) {
    // Weight by inverse distance (closer neighbors have higher weight)
    torch::Tensor weights = 1.0 / (distances + 1e-8);
    torch::Tensor weight_sum = torch::sum(weights, 1, true);
    
    // Normalize weights
    weights = weights / weight_sum;
    
    // Compute weighted mean
    return torch::sum(distances * weights, 1);
}

} // namespace utils
} // namespace patchcore

