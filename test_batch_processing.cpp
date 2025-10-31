#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>

int main() {
    try {
        // Load the model
        torch::jit::script::Module model = torch::jit::load("models/patchcore_extractor.pt");
        model.eval();
        
        // Load multiple test images
        std::vector<std::string> image_paths = {
            "D:/PROJECTS/1.Datasets/5.Anomalib_datasets/MVTecAD/bottle/train/good/000.png",
            "D:/PROJECTS/1.Datasets/5.Anomalib_datasets/MVTecAD/bottle/train/good/001.png",
            "D:/PROJECTS/1.Datasets/5.Anomalib_datasets/MVTecAD/bottle/train/good/002.png"
        };
        
        std::vector<cv::Mat> images;
        for (const auto& path : image_paths) {
            cv::Mat image = cv::imread(path);
            if (image.empty()) {
                std::cerr << "Could not load image: " << path << std::endl;
                continue;
            }
            images.push_back(image);
        }
        
        std::cout << "Loaded " << images.size() << " images" << std::endl;
        
        // Process images one by one (like in the fixed extractFeaturesBatch)
        std::vector<torch::Tensor> all_features;
        
        for (size_t i = 0; i < images.size(); ++i) {
            std::cout << "Processing image " << (i + 1) << "/" << images.size() << std::endl;
            
            // Resize to 256x256
            cv::Mat resized;
            cv::resize(images[i], resized, cv::Size(256, 256));
            
            // Convert to tensor
            torch::Tensor tensor = torch::from_blob(
                resized.data, 
                {resized.rows, resized.cols, resized.channels()}, 
                torch::kFloat
            );
            
            // Permute dimensions from HWC to CHW
            tensor = tensor.permute({2, 0, 1});
            
            // Add batch dimension
            tensor = tensor.unsqueeze(0);
            
            // Normalize with ImageNet statistics
            torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1});
            torch::Tensor std = torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1});
            tensor = (tensor / 255.0 - mean) / std;
            
            // Forward pass through model
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(tensor);
            
            torch::Tensor output = model.forward(inputs).toTensor();
            std::cout << "  Model output shape: " << output.sizes() << std::endl;
            
            // Apply patchifyFeatures
            int patch_size = 3;
            int padding = patch_size / 2;
            torch::Tensor patched = torch::avg_pool2d(output, patch_size, 1, padding, false);
            std::cout << "  Patched features shape: " << patched.sizes() << std::endl;
            
            // Reshape for memory bank
            int batch_size = patched.size(0);
            int channels = patched.size(1);
            int height = patched.size(2);
            int width = patched.size(3);
            
            torch::Tensor features_flat = patched.view({batch_size, channels, height * width});
            features_flat = features_flat.permute({0, 2, 1}).contiguous();
            features_flat = features_flat.view({-1, channels});
            
            std::cout << "  Features flat shape: " << features_flat.sizes() << std::endl;
            
            all_features.push_back(features_flat);
        }
        
        // Concatenate all features
        if (!all_features.empty()) {
            torch::Tensor combined_features = torch::cat(all_features, 0);
            std::cout << "Combined features shape: " << combined_features.sizes() << std::endl;
        }
        
        std::cout << "Batch processing test successful!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

