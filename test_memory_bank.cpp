#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        // Load the model
        torch::jit::script::Module model = torch::jit::load("models/patchcore_extractor.pt");
        model.eval();
        
        // Create test features (simulating what we get from the model)
        torch::Tensor test_features = torch::randn({1, 384, 32, 32});
        std::cout << "Test features shape: " << test_features.sizes() << std::endl;
        
        // Test reshaping to [N, C] format
        int batch_size = test_features.size(0);
        int channels = test_features.size(1);
        int height = test_features.size(2);
        int width = test_features.size(3);
        
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Channels: " << channels << std::endl;
        std::cout << "Height: " << height << std::endl;
        std::cout << "Width: " << width << std::endl;
        
        // Reshape to [N, C*H*W] then to [N*H*W, C]
        torch::Tensor reshaped = test_features.view({batch_size, -1}); // [1, 384*32*32]
        std::cout << "Reshaped shape: " << reshaped.sizes() << std::endl;
        
        // Reshape to [N*H*W, C] for memory bank
        torch::Tensor features_flat = reshaped.view({-1, channels}); // [1*32*32, 384]
        std::cout << "Features flat shape: " << features_flat.sizes() << std::endl;
        
        // Test distance computation
        std::cout << "Testing distance computation..." << std::endl;
        torch::Tensor features1 = features_flat.slice(0, 0, 100); // Take first 100 features
        torch::Tensor features2 = features_flat.slice(0, 100, 200); // Take next 100 features
        
        std::cout << "Features1 shape: " << features1.sizes() << std::endl;
        std::cout << "Features2 shape: " << features2.sizes() << std::endl;
        
        // Compute L2 distance
        torch::Tensor distances = torch::cdist(features1, features2, 2);
        std::cout << "Distances shape: " << distances.sizes() << std::endl;
        
        std::cout << "Test successful!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

