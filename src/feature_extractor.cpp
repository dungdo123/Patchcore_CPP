#include "patchcore.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>

namespace patchcore {

FeatureExtractor::FeatureExtractor(const PatchCoreConfig& config) 
    : config_(config), device_(config.use_gpu ? torch::kCUDA : torch::kCPU) {
}

bool FeatureExtractor::loadModel(const std::string& model_path) {
    try {
        std::cout << "Loading model from: " << model_path << std::endl;
        std::cout << "Expected model architecture: WideResNet50 (layer2+layer3) with projection to " 
                  << config_.target_dim << " dimensions" << std::endl;
        
        // Load the TorchScript model
        // Note: The model should already include:
        // 1. Feature extraction from WideResNet50 layer2 and layer3
        // 2. Upsampling and concatenation of layer2 and layer3 features
        // 3. Projection to target_dim using 1x1 convolution
        auto loaded_model = torch::jit::load(model_path);
        model_ = std::make_shared<torch::jit::script::Module>(loaded_model);
        model_->eval();
        
        // Move model to the appropriate device
        if (device_.is_cuda()) {
            model_->to(torch::kCUDA);
        }
        
        std::cout << "Model loaded successfully from: " << model_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat FeatureExtractor::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    
    // Resize to target size
    cv::resize(image, processed, cv::Size(config_.input_size, config_.input_size));
    
    // Convert BGR to RGB
    cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    
    // Convert to float and normalize
    processed.convertTo(processed, CV_32F, 1.0 / 255.0);
    
    return processed;
}

torch::Tensor FeatureExtractor::imageToTensor(const cv::Mat& image) {
    // Convert OpenCV Mat to torch Tensor
    // Create tensor with shape [height, width, channels]
    torch::Tensor tensor = torch::from_blob(
        image.data, 
        {image.rows, image.cols, image.channels()}, 
        torch::kFloat
    );
    
    // Permute dimensions from HWC to CHW and add batch dimension
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0); // [1, channels, height, width]
    
    // Normalize with ImageNet statistics
    tensor = utils::normalizeTensor(tensor, config_.mean, config_.std);
    
    return tensor.to(device_);
}

torch::Tensor FeatureExtractor::extractFeatures(const cv::Mat& image) {
    if (!model_) {
        throw std::runtime_error("Model not loaded");
    }
    
    // Preprocess image
    cv::Mat processed = preprocessImage(image);
    
    // Convert to tensor
    torch::Tensor input_tensor = imageToTensor(processed);
    
    std::cout << "Input tensor shape: [" << input_tensor.size(0) << ", " << input_tensor.size(1) << ", " << input_tensor.size(2) << ", " << input_tensor.size(3) << "]" << std::endl;
    
    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);
    
    torch::Tensor output;
    try {
        output = model_->forward(inputs).toTensor();
        std::cout << "Model output shape: [" << output.size(0) << ", " << output.size(1) << ", " << output.size(2) << ", " << output.size(3) << "]" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error during model inference: " << e.what() << std::endl;
        throw;
    }
    
    // Validate model output dimensions
    if (output.size(1) == 3) {
        std::cerr << "Error: Model returned input image instead of features. Shape: [" 
                  << output.size(0) << ", " << output.size(1) << ", " 
                  << output.size(2) << ", " << output.size(3) << "]" << std::endl;
        std::cerr << "This suggests the model file may be corrupted or incorrectly created." << std::endl;
        throw std::runtime_error("Model returned image tensor instead of features");
    }
    
    // Verify output dimension matches expected target_dim
    if (output.size(1) != config_.target_dim) {
        std::cerr << "Warning: Model output dimension (" << output.size(1) 
                  << ") doesn't match expected target_dim (" << config_.target_dim << ")" << std::endl;
        std::cerr << "This may indicate a model architecture mismatch." << std::endl;
        // Continue anyway - might still work if dimensions are compatible
    } else {
        std::cout << "Model output dimension verified: " << output.size(1) << " (matches target_dim)" << std::endl;
    }
    
    std::cout << "Model output spatial size: " << output.size(2) << "x" << output.size(3) << std::endl;
    
    // Apply patchification (local aggregation with 3x3 average pooling)
    // Note: The model already includes feature projection to target_dim
    // Patchification is applied AFTER projection in the official PatchCore pipeline
    // Patchification with 3x3 avg pool (padding=1, stride=1) preserves spatial dimensions
    torch::Tensor patched_features = patchifyFeatures(output);
    
    std::cout << "Patched features spatial size: " << patched_features.size(2) << "x" << patched_features.size(3) << std::endl;
    
    return patched_features;
}

torch::Tensor FeatureExtractor::extractFeaturesBatch(const std::vector<cv::Mat>& images) {
    if (!model_) {
        throw std::runtime_error("Model not loaded");
    }
    
    std::vector<torch::Tensor> all_features;
    
    // Process each image individually since the model was traced with single image input
    for (const auto& image : images) {
        cv::Mat processed = preprocessImage(image);
        torch::Tensor tensor = imageToTensor(processed);
        
        std::cout << "Batch input tensor shape: [" << tensor.size(0) << ", " << tensor.size(1) << ", " << tensor.size(2) << ", " << tensor.size(3) << "]" << std::endl;
        
        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);
        
        torch::Tensor output;
        try {
            output = model_->forward(inputs).toTensor();
            std::cout << "Model output shape: [" << output.size(0) << ", " << output.size(1) << ", " << output.size(2) << ", " << output.size(3) << "]" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error during model inference: " << e.what() << std::endl;
            throw;
        }
        
        // Validate model output dimensions
        if (output.size(1) == 3) {
            std::cerr << "Error: Model returned input image instead of features. Shape: [" 
                      << output.size(0) << ", " << output.size(1) << ", " 
                      << output.size(2) << ", " << output.size(3) << "]" << std::endl;
            std::cerr << "This suggests the model file may be corrupted or incorrectly created." << std::endl;
            throw std::runtime_error("Model returned image tensor instead of features");
        }
        
        // Verify output dimension matches expected target_dim
        if (output.size(1) != config_.target_dim) {
            std::cerr << "Warning: Model output dimension (" << output.size(1) 
                      << ") doesn't match expected target_dim (" << config_.target_dim << ")" << std::endl;
            std::cerr << "This may indicate a model architecture mismatch." << std::endl;
            // Continue anyway - might still work if dimensions are compatible
        }
        
        // Apply patchification (local aggregation with 3x3 average pooling)
        // Note: The model already includes feature projection to target_dim
        // Patchification is applied AFTER projection in the official PatchCore pipeline
        torch::Tensor patched_features = patchifyFeatures(output);
        
        all_features.push_back(patched_features);
    }
    
    // Stack all features into a batch
    std::cout << "Stacking " << all_features.size() << " features..." << std::endl;
    
    if (all_features.empty()) {
        std::cerr << "Error: No features to stack" << std::endl;
        throw std::runtime_error("No features to stack");
    }
    
    std::cout << "First feature shape: [" << all_features[0].size(0) << ", " << all_features[0].size(1) << ", " << all_features[0].size(2) << ", " << all_features[0].size(3) << "]" << std::endl;
    
    return torch::stack(all_features);
}

torch::Tensor FeatureExtractor::extractLayerFeatures(torch::Tensor input, const std::string& layer_name) {
    // This is a simplified version - in practice, you'd need to modify the model
    // to return intermediate layer outputs or use hooks
    
    // For now, we'll assume the model returns features from the specified layers
    // You may need to modify your PyTorch model to support this
    
    // Placeholder implementation
    return input;
}

torch::Tensor FeatureExtractor::upsampleToTargetResolution(torch::Tensor features, int target_size) {
    // Upsample features to target resolution using bilinear interpolation
    int current_height = features.size(2);
    int current_width = features.size(3);
    
    if (current_height == target_size && current_width == target_size) {
        return features;  // Already at target resolution
    }
    
    // Use bilinear interpolation to upsample
    torch::Tensor upsampled = torch::nn::functional::interpolate(
        features,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{target_size, target_size})
            .mode(torch::kBilinear)
            .align_corners(false)
    );
    
    return upsampled;
}

torch::Tensor FeatureExtractor::patchifyFeatures(torch::Tensor features) {
    // Check if the input is an image tensor instead of features
    if (features.size(1) == 3) {
        std::cerr << "Error: patchifyFeatures received image tensor instead of features. Shape: [" 
                  << features.size(0) << ", " << features.size(1) << ", " 
                  << features.size(2) << ", " << features.size(3) << "]" << std::endl;
        throw std::runtime_error("patchifyFeatures received image tensor instead of features");
    }
    
    // Apply local patch aggregation (3x3 average pooling)
    int patch_size = config_.patch_size;
    int padding = patch_size / 2;
    
    torch::Tensor patched = torch::avg_pool2d(
        features, 
        patch_size, 
        1, 
        padding, 
        false
    );
    
    return patched;
}

torch::Tensor FeatureExtractor::projectFeatures(torch::Tensor features) {
    // NOTE: This function is DEPRECATED with the new model architecture.
    // The new model (v2) already includes projection in the TorchScript model.
    // This function is kept for backward compatibility only.
    // 
    // Project features to target dimension (legacy implementation)
    int current_dim = features.size(1);
    int target_dim = config_.target_dim;
    
    if (current_dim == target_dim) {
        return features;
    }
    
    // Create projection layer
    torch::nn::Linear projection(current_dim, target_dim);
    projection->to(device_);
    
    // Initialize weights
    torch::nn::init::xavier_uniform_(projection->weight);
    torch::nn::init::zeros_(projection->bias);
    
    // Apply projection
    torch::Tensor projected = projection->forward(features);
    
    return projected;
}

FeatureExtractor::ModelInfo FeatureExtractor::getModelInfo() const {
    ModelInfo info;
    
    if (model_) {
        // Get model parameters info
        auto parameters = model_->parameters();
        int param_count = 0;
        
        for (const auto& param : parameters) {
            param_count += param.numel();
        }
        
        info.parameters["total_parameters"] = {param_count};
        info.parameters["device"] = {device_.is_cuda() ? 1 : 0};
    }
    
    return info;
}

} // namespace patchcore
