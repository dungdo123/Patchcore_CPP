#include "patchcore.h"
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#include <windows.h>
#endif

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
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <model_path> <normal_images_dir> <test_images_dir> [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --metric cosine|l2     Distance metric (default: l2)" << std::endl;
        std::cout << "  --core-set-size N     Core set size (default: 10000)" << std::endl;
        std::cout << "  --k-neighbors N       Number of neighbors (default: 1)" << std::endl;
        std::cout << "  --aggregation method  Aggregation method: mean|max|median|weighted_mean (default: mean)" << std::endl;
        std::cout << "  --output-dir path     Output directory for results (default: ./output)" << std::endl;
        std::cout << "  --save-memory-bank path  Save memory bank after training (optional)" << std::endl;
        std::cout << "  --load-memory-bank path   Load saved memory bank instead of training (optional)" << std::endl;
        std::cout << "  --gpu                 Use GPU (default: true)" << std::endl;
        std::cout << "  --cpu                 Use CPU instead of GPU" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string normal_images_dir = argv[2];
    std::string test_images_dir = argv[3];
    
    // Parse command line arguments
    patchcore::DistanceMetric metric = patchcore::DistanceMetric::L2;
    int core_set_size = 10000;
    int k_neighbors = 1;
    std::string aggregation = "mean";
    std::string output_dir = "./output";
    std::string save_memory_bank_path;
    std::string load_memory_bank_path;
    bool use_gpu = true;
    
    for (int i = 4; i < argc; ++i) {
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
        } else if (arg == "--core-set-size" && i + 1 < argc) {
            core_set_size = std::stoi(argv[++i]);
        } else if (arg == "--k-neighbors" && i + 1 < argc) {
            k_neighbors = std::stoi(argv[++i]);
        } else if (arg == "--aggregation" && i + 1 < argc) {
            aggregation = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--save-memory-bank" && i + 1 < argc) {
            save_memory_bank_path = argv[++i];
        } else if (arg == "--load-memory-bank" && i + 1 < argc) {
            load_memory_bank_path = argv[++i];
        } else if (arg == "--cpu") {
            use_gpu = false;
        } else if (arg == "--gpu") {
            use_gpu = true;
        }
    }
    
    // Create configuration
    patchcore::PatchCoreConfig config;
    config.model_path = model_path;
    config.layers_to_extract = {}; // Model handles layer extraction internally
    config.patch_size = 3;
    config.target_dim = 384;
    config.core_set_size = core_set_size;
    config.metric = metric;
    config.auto_size = true;
    config.coverage_threshold = 0.95;
    config.input_size = 256;
    config.k_neighbors = k_neighbors;
    config.aggregation = aggregation;
    config.use_gpu = use_gpu;
    
    std::cout << "PatchCore Configuration:" << std::endl;
    std::cout << "  Model path: " << config.model_path << std::endl;
    std::cout << "  Normal images: " << normal_images_dir << std::endl;
    std::cout << "  Test images: " << test_images_dir << std::endl;
    std::cout << "  Distance metric: " << (metric == patchcore::DistanceMetric::COSINE ? "cosine" : "l2") << std::endl;
    std::cout << "  Core set size: " << config.core_set_size << std::endl;
    std::cout << "  K neighbors: " << config.k_neighbors << std::endl;
    std::cout << "  Aggregation: " << config.aggregation << std::endl;
    std::cout << "  Device: " << (config.use_gpu ? "GPU" : "CPU") << std::endl;
    std::cout << "  Output directory: " << output_dir << std::endl;
    if (!load_memory_bank_path.empty()) {
        std::cout << "  Load memory bank: " << load_memory_bank_path << std::endl;
    }
    if (!save_memory_bank_path.empty()) {
        std::cout << "  Save memory bank: " << save_memory_bank_path << std::endl;
    }
    
    try {
        // Create detector
        patchcore::PatchCoreDetector detector(config);
        
        // Initialize detector
        if (!detector.initialize()) {
            std::cerr << "Failed to initialize detector" << std::endl;
            return 1;
        }
        
        // Load or train memory bank
        if (!load_memory_bank_path.empty()) {
            // Load saved memory bank
            std::cout << "\nLoading saved memory bank from: " << load_memory_bank_path << std::endl;
            try {
                detector.loadMemoryBank(load_memory_bank_path);
                std::cout << "Memory bank loaded successfully. Skipping training." << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to load memory bank: " << e.what() << std::endl;
                std::cerr << "Falling back to training from normal images..." << std::endl;
                // Fall through to training
                load_memory_bank_path.clear(); // Clear to proceed with training
            }
        }
        
        if (load_memory_bank_path.empty()) {
            // Train detector from normal images
            std::cout << "\nLoading normal images..." << std::endl;
            std::vector<cv::Mat> normal_images = patchcore::utils::loadImagesFromDirectory(normal_images_dir);
            if (normal_images.empty()) {
                std::cerr << "No normal images found in: " << normal_images_dir << std::endl;
                return 1;
            }
            std::cout << "Loaded " << normal_images.size() << " normal images" << std::endl;
            
            // Train detector
            std::cout << "\nTraining detector..." << std::endl;
            detector.train(normal_images);
            
            // Save memory bank if requested
            if (!save_memory_bank_path.empty()) {
                std::cout << "\nSaving memory bank to: " << save_memory_bank_path << std::endl;
                try {
                    detector.saveMemoryBank(save_memory_bank_path);
                    std::cout << "Memory bank saved successfully!" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to save memory bank: " << e.what() << std::endl;
                }
            }
        }
        
        // Load test images
        std::cout << "\nLoading test images..." << std::endl;
        std::vector<cv::Mat> test_images = patchcore::utils::loadImagesFromDirectory(test_images_dir);
        if (test_images.empty()) {
            std::cerr << "No test images found in: " << test_images_dir << std::endl;
            return 1;
        }
        std::cout << "Loaded " << test_images.size() << " test images" << std::endl;
        
        // Predict anomalies
        std::cout << "\nPredicting anomalies..." << std::endl;
        auto result = detector.predict(test_images);
        
        // Print results
        std::cout << "\nAnomaly Detection Results:" << std::endl;
        std::cout << "=========================" << std::endl;
        
        for (int i = 0; i < result.image_scores.size(0); ++i) {
            float score = result.image_scores[i].item<float>();
            std::cout << "Image " << i + 1 << ": Anomaly Score = " << score << std::endl;
        }
        
        // Save results
        std::cout << "\nSaving results to: " << output_dir << std::endl;
        
        // Create output directory
        std::filesystem::create_directories(output_dir);
        
        // Save anomaly maps
        for (int i = 0; i < result.anomaly_maps.size(); ++i) {
            std::string output_path = output_dir + "/anomaly_map_" + std::to_string(i + 1) + ".jpg";
            cv::imwrite(output_path, result.anomaly_maps[i]);
        }
        
        // Save scores to file
        std::string scores_file = output_dir + "/anomaly_scores.txt";
        std::ofstream scores_out(scores_file);
        if (scores_out.is_open()) {
            for (int i = 0; i < result.image_scores.size(0); ++i) {
                float score = result.image_scores[i].item<float>();
                scores_out << "Image " << i + 1 << ": " << score << std::endl;
            }
            scores_out.close();
        }
        
        // Get detector info
        auto info = detector.getInfo();
        std::cout << "\nDetector Information:" << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "Memory bank size: " << info.memory_bank_info.core_set_size << std::endl;
        std::cout << "Feature dimension: " << info.memory_bank_info.feature_dim << std::endl;
        std::cout << "Distance metric: " << info.memory_bank_info.metric << std::endl;
        
        std::cout << "\nPatchCore inference completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

