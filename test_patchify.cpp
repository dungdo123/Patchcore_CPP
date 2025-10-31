#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        // Load the model
        torch::jit::script::Module model = torch::jit::load("models/patchcore_extractor.pt");
        model.eval();
        
        // Create a test input
        torch::Tensor test_input = torch::randn({1, 3, 256, 256});
        std::cout << "Input shape: " << test_input.sizes() << std::endl;
        
        // Forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(test_input);
        
        torch::Tensor output = model.forward(inputs).toTensor();
        std::cout << "Model output shape: " << output.sizes() << std::endl;
        
        // Test patchifyFeatures function
        int patch_size = 3;
        int padding = patch_size / 2;
        
        std::cout << "Testing patchifyFeatures..." << std::endl;
        torch::Tensor patched = torch::avg_pool2d(
            output, 
            patch_size, 
            1, 
            padding, 
            false
        );
        
        std::cout << "Patched output shape: " << patched.sizes() << std::endl;
        std::cout << "Test successful!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

