#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        // Load the model
        torch::jit::script::Module model = torch::jit::load("models/patchcore_extractor.pt");
        model.eval();
        
        // Create a simple test input
        torch::Tensor test_input = torch::randn({1, 3, 256, 256});
        std::cout << "Input shape: " << test_input.sizes() << std::endl;
        
        // Forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(test_input);
        
        torch::Tensor output = model.forward(inputs).toTensor();
        std::cout << "Output shape: " << output.sizes() << std::endl;
        std::cout << "Model test successful!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

