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
        
        // Load a test image
        std::string image_path = "D:/PROJECTS/1.Datasets/5.Anomalib_datasets/MVTecAD/bottle/train/good/000.png";
        cv::Mat image = cv::imread(image_path);
        
        if (image.empty()) {
            std::cerr << "Could not load image: " << image_path << std::endl;
            return 1;
        }
        
        std::cout << "Original image shape: " << image.rows << "x" << image.cols << "x" << image.channels() << std::endl;
        
        // Resize to 256x256
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(256, 256));
        std::cout << "Resized image shape: " << resized.rows << "x" << resized.cols << "x" << resized.channels() << std::endl;
        
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
        
        std::cout << "Tensor shape after preprocessing: " << tensor.sizes() << std::endl;
        
        // Normalize with ImageNet statistics
        torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1});
        torch::Tensor std = torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1});
        tensor = (tensor / 255.0 - mean) / std;
        
        std::cout << "Tensor shape after normalization: " << tensor.sizes() << std::endl;
        
        // Forward pass through model
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(tensor);
        
        std::cout << "Running model forward pass..." << std::endl;
        torch::Tensor output = model.forward(inputs).toTensor();
        std::cout << "Model output shape: " << output.sizes() << std::endl;
        
        // Test patchifyFeatures function
        int patch_size = 3;
        int padding = patch_size / 2;
        
        std::cout << "Testing patchifyFeatures..." << std::endl;
        torch::Tensor patched = torch::avg_pool2d(output, patch_size, 1, padding, false);
        std::cout << "Patched features shape: " << patched.sizes() << std::endl;
        
        // Test reshaping for memory bank
        int batch_size = patched.size(0);
        int channels = patched.size(1);
        int height = patched.size(2);
        int width = patched.size(3);
        
        torch::Tensor features_flat = patched.view({batch_size, channels, height * width});
        features_flat = features_flat.permute({0, 2, 1}).contiguous();
        features_flat = features_flat.view({-1, channels});
        
        std::cout << "Features flat shape: " << features_flat.sizes() << std::endl;
        
        std::cout << "Full flow test successful!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

