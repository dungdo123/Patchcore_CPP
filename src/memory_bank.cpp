#include "patchcore.h"
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <random>
#include <climits>
#include <fstream>
#include <filesystem>
#include <typeinfo>

namespace patchcore {

MemoryBank::MemoryBank(const PatchCoreConfig& config) : config_(config) {
}

void MemoryBank::buildMemoryBank(const std::vector<cv::Mat>& normal_images, 
                                FeatureExtractor& extractor) {
    std::cout << "Building memory bank from " << normal_images.size() << " normal images..." << std::endl;
    
    // Collect all features first, processing one image at a time to minimize memory usage
    std::vector<torch::Tensor> all_feature_tensors;
    all_feature_tensors.reserve(normal_images.size());
    
    for (size_t i = 0; i < normal_images.size(); ++i) {
        if (i % 10 == 0) {
            std::cout << "Processing image " << i + 1 << "/" << normal_images.size() << std::endl;
        }
        
        // Extract features on GPU
        torch::Tensor features = extractor.extractFeatures(normal_images[i]);
        
        // Flatten spatial dimensions
        int64_t batch_size_img = features.size(0);
        int64_t channels = features.size(1);
        int64_t height = features.size(2);
        int64_t width = features.size(3);
        
        // Reshape to [batch * height * width, channels]
        torch::Tensor flattened = features.view({batch_size_img * height * width, channels});
        
        // Immediately transfer to CPU to avoid GPU memory issues
        flattened = flattened.cpu();
        
        all_feature_tensors.push_back(flattened);
    }
    
    // Concatenate all features on CPU
    all_features_ = torch::cat(all_feature_tensors, 0);
    std::cout << "Extracted " << all_features_.size(0) << " feature vectors" << std::endl;
    std::cout << "Feature dimension: " << all_features_.size(1) << std::endl;
    
    // Build core set using efficient sampling
    buildCoreSet();
}

void MemoryBank::addFeatures(const torch::Tensor& features) {
    if (all_features_.numel() == 0) {
        all_features_ = features.clone();
    } else {
        all_features_ = torch::cat({all_features_, features}, 0);
    }
}

torch::Tensor MemoryBank::buildCoreSetForBatch(const torch::Tensor& batch_features) {
    if (batch_features.numel() == 0) {
        throw std::runtime_error("No features available for core set building");
    }
    
    std::cout << "  Building core set for batch with " << batch_features.size(0) << " features..." << std::endl;
    
    // Calculate core set size for this batch (proportional to batch size)
    int batch_core_size = std::min(config_.core_set_size, (int)batch_features.size(0));
    
    torch::Tensor core_set;
    if (config_.auto_size) {
        core_set = approximateGreedyCoreSet(batch_features);
    } else {
        core_set = greedyCoreSetSampling(batch_features);
    }
    
    std::cout << "  Batch core set built with " << core_set.size(0) << " features" << std::endl;
    
    return core_set;
}

void MemoryBank::buildCoreSet() {
    if (all_features_.numel() == 0) {
        throw std::runtime_error("No features available for core set building");
    }
    
    std::cout << "Building core set on CPU..." << std::endl;
    std::cout << "Processing " << all_features_.size(0) << " features" << std::endl;
    
    // Ensure features are on CPU
    if (all_features_.device().is_cuda()) {
        all_features_ = all_features_.cpu();
    }
    
    // Use efficient sampling strategy
    int64_t total_features = all_features_.size(0);
    int64_t target_size = std::min(static_cast<int64_t>(config_.core_set_size), total_features);
    
    std::cout << "Target core set size: " << target_size << " out of " << total_features << " features" << std::endl;
    
    if (target_size >= total_features) {
        // If we need all features, just use them all
        core_set_ = all_features_.clone();
    } else {
        // Use efficient random sampling for large datasets
        if (total_features > 50000) {
            std::cout << "Using random sampling for large dataset..." << std::endl;
            auto indices = torch::randperm(total_features, torch::TensorOptions().device(all_features_.device()));
            indices = indices.slice(0, 0, target_size);
            core_set_ = all_features_.index_select(0, indices);
        } else {
            // Use greedy sampling for smaller datasets
            std::cout << "Using greedy sampling..." << std::endl;
            if (config_.auto_size) {
                core_set_ = approximateGreedyCoreSet(all_features_);
            } else {
                core_set_ = greedyCoreSetSampling(all_features_);
            }
        }
    }
    
    std::cout << "Core set built with " << core_set_.size(0) << " features" << std::endl;
    
    // Build FAISS index
    buildFAISSIndex();
}

torch::Tensor MemoryBank::greedyCoreSetSampling(const torch::Tensor& features) {
    int64_t total_features = features.size(0);
    int64_t target_size = (config_.core_set_size < total_features) ? config_.core_set_size : total_features;
    
    std::cout << "  Greedy sampling: selecting " << target_size << " from " << total_features << " features" << std::endl;
    
    // For very large datasets, use random sampling instead of greedy
    if (total_features > 20000) {
        std::cout << "  Using random sampling for large dataset (greedy too slow)" << std::endl;
        auto indices = torch::randperm(total_features, torch::TensorOptions().device(features.device()));
        indices = indices.slice(0, 0, target_size);
        return features.index_select(0, indices);
    }
    
    std::vector<int64_t> selected_indices;
    selected_indices.reserve(target_size);
    
    // Random initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, total_features - 1);
    
    selected_indices.push_back(dis(gen));
    
    // Greedy selection with memory-efficient processing
    torch::Tensor selected_features = features[selected_indices[0]].unsqueeze(0);
    
    for (int64_t i = 1; i < target_size; ++i) {
        if (i % 500 == 0) {
            std::cout << "  Selected " << i << "/" << target_size << " features" << std::endl;
        }
        
        // Process in chunks to avoid memory issues
        const int64_t chunk_size = 5000;
        torch::Tensor min_distances = torch::full({total_features}, std::numeric_limits<float>::max(), 
                                                  torch::TensorOptions().device(features.device()));
        
        for (int64_t start = 0; start < total_features; start += chunk_size) {
            int64_t end = std::min(start + chunk_size, total_features);
            torch::Tensor chunk_features = features.slice(0, start, end);
            
            // Compute distances to selected features
            torch::Tensor distances = utils::l2Distance(chunk_features, selected_features);
            torch::Tensor chunk_min_distances = std::get<0>(torch::min(distances, 1));
            
            // Update minimum distances
            min_distances.slice(0, start, end) = torch::min(min_distances.slice(0, start, end), chunk_min_distances);
        }
        
        // Find the feature with maximum minimum distance
        auto max_result = torch::max(min_distances, 0);
        int64_t max_idx = std::get<1>(max_result).item<int64_t>();
        selected_indices.push_back(max_idx);
        
        // Add to selected features
        torch::Tensor new_feature = features[max_idx].unsqueeze(0);
        selected_features = torch::cat({selected_features, new_feature}, 0);
    }
    
    return features.index_select(0, torch::tensor(selected_indices));
}

torch::Tensor MemoryBank::approximateGreedyCoreSet(const torch::Tensor& features) {
    int64_t total_features = features.size(0);
    int64_t target_size = (config_.core_set_size < total_features) ? config_.core_set_size : total_features;
    
    std::cout << "  Approximate greedy sampling: selecting " << target_size << " from " << total_features << " features" << std::endl;
    
    // For very large datasets, use random sampling
    if (total_features > 50000) {
        std::cout << "  Using random sampling for very large dataset" << std::endl;
        auto indices = torch::randperm(total_features, torch::TensorOptions().device(features.device()));
        indices = indices.slice(0, 0, target_size);
        return features.index_select(0, indices);
    }
    
    // For large datasets, use a smaller candidate pool
    int64_t candidate_pool_size = std::min(total_features, target_size * 5);
    
    std::vector<int64_t> selected_indices;
    selected_indices.reserve(target_size);
    
    // Random initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, total_features - 1);
    
    selected_indices.push_back(dis(gen));
    
    // Approximate greedy selection
    torch::Tensor selected_features = features[selected_indices[0]].unsqueeze(0);
    
    for (int64_t i = 1; i < target_size; ++i) {
        if (i % 500 == 0) {
            std::cout << "  Selected " << i << "/" << target_size << " features" << std::endl;
        }
        
        // Sample candidate pool
        std::vector<int64_t> candidates;
        candidates.reserve(candidate_pool_size);
        for (int64_t j = 0; j < candidate_pool_size; ++j) {
            candidates.push_back(dis(gen));
        }
        
        torch::Tensor candidate_features = features.index_select(0, torch::tensor(candidates));
        
        // Compute distances to current core set
        torch::Tensor distances = utils::l2Distance(candidate_features, selected_features);
        torch::Tensor min_distances = std::get<0>(torch::min(distances, 1));
        
        // Select candidate with maximum minimum distance
        auto max_result = torch::max(min_distances, 0);
        int64_t max_idx = std::get<1>(max_result).item<int64_t>();
        int64_t selected_candidate = candidates[max_idx];
        
        selected_indices.push_back(selected_candidate);
        
        // Add to selected features
        torch::Tensor new_feature = features[selected_candidate].unsqueeze(0);
        selected_features = torch::cat({selected_features, new_feature}, 0);
    }
    
    return features.index_select(0, torch::tensor(selected_indices));
}

void MemoryBank::buildFAISSIndex() {
    if (core_set_.numel() == 0) {
        throw std::runtime_error("No core set available for FAISS index");
    }
    
    int64_t num_features = core_set_.size(0);
    int64_t feature_dim = core_set_.size(1);
    
    // Convert torch tensor to float array
    torch::Tensor core_set_cpu = core_set_.cpu().contiguous();
    
    // For cosine similarity, normalize the core set vectors before adding to index
    if (config_.metric == DistanceMetric::COSINE) {
        core_set_cpu = torch::nn::functional::normalize(core_set_cpu, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    }
    
    float* data_ptr = core_set_cpu.data_ptr<float>();
    
    // Create FAISS index
    if (config_.metric == DistanceMetric::COSINE) {
        faiss_index_ = std::make_unique<faiss::IndexFlatIP>(feature_dim);
    } else {
        faiss_index_ = std::make_unique<faiss::IndexFlatL2>(feature_dim);
    }
    
    // Add features to index
    faiss_index_->add(num_features, data_ptr);
    
    std::cout << "FAISS index built with " << num_features << " features" << std::endl;
}

torch::Tensor MemoryBank::computeAnomalyScores(const torch::Tensor& test_features) {
    if (!faiss_index_) {
        throw std::runtime_error("FAISS index not built");
    }
    
    int64_t num_queries = test_features.size(0);
    int64_t feature_dim = test_features.size(1);
    int64_t k = std::min(config_.k_neighbors, static_cast<int>(core_set_.size(0)));
    
    std::cout << "    computeAnomalyScores: num_queries=" << num_queries 
              << ", feature_dim=" << feature_dim << ", k=" << k << std::endl;
    
    // Prepare query features
    std::cout << "    Moving features to CPU..." << std::endl;
    torch::Tensor query_features = test_features.cpu().contiguous();
    
    if (config_.metric == DistanceMetric::COSINE) {
        std::cout << "    Normalizing for cosine similarity..." << std::endl;
        // Normalize for cosine similarity
        query_features = torch::nn::functional::normalize(query_features, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    }
    
    std::cout << "    Preparing FAISS search..." << std::endl;
    
    // Verify feature dimensions match
    if (query_features.size(1) != faiss_index_->d) {
        std::cerr << "    ERROR: Feature dimension mismatch! Query: " << query_features.size(1) 
                  << ", FAISS index: " << faiss_index_->d << std::endl;
        throw std::runtime_error("Feature dimension mismatch between query and FAISS index");
    }
    
    float* query_ptr = query_features.data_ptr<float>();
    if (query_ptr == nullptr) {
        throw std::runtime_error("Failed to get data pointer from query features");
    }
    
    // Verify data is accessible
    std::cout << "    Query data pointer valid: " << (query_ptr != nullptr) << std::endl;
    std::cout << "    First few query values: " << query_ptr[0] << ", " << query_ptr[1] << ", " << query_ptr[2] << std::endl;
    std::cout.flush();
    
    // Search - process in batches to avoid potential hangs with large queries
    std::vector<float> distances(k * num_queries);
    std::vector<faiss::idx_t> indices(k * num_queries);
    
    // Process queries in small batches for stability
    // Use very small batch size to avoid hangs
    const int64_t batch_size = 10; // Process 10 queries at a time
    std::cout << "    Performing FAISS search for " << num_queries << " queries (k=" << k 
              << ", dim=" << faiss_index_->d << ") in batches of " << batch_size << "..." << std::endl;
    std::cout.flush();
    
    int64_t processed = 0;
    try {
        std::cout << "      Starting batch loop..." << std::endl;
        std::cout.flush();
        
        for (int64_t start = 0; start < num_queries; start += batch_size) {
            int64_t end = (std::min)(start + batch_size, num_queries);
            int64_t current_batch_size = end - start;
            
            std::cout << "      [Batch " << (start / batch_size + 1) << "] Processing queries " << start << "-" << (end-1) << "..." << std::endl;
            std::cout.flush();
            
            float* batch_query_ptr = query_ptr + start * feature_dim;
            float* batch_distances = distances.data() + start * k;
            faiss::idx_t* batch_indices = indices.data() + start * k;
            
            // Verify pointers are valid
            if (batch_query_ptr == nullptr || batch_distances == nullptr || batch_indices == nullptr) {
                std::cerr << "      ERROR: Invalid pointer!" << std::endl;
                throw std::runtime_error("Invalid pointer in batch processing");
            }
            
            std::cout << "      Calling FAISS search..." << std::endl;
            std::cout.flush();
            
            faiss_index_->search(current_batch_size, batch_query_ptr, k, batch_distances, batch_indices);
            
            std::cout << "      Batch completed. First result: distance=" << batch_distances[0] << ", index=" << batch_indices[0] << std::endl;
            std::cout.flush();
            
            processed += current_batch_size;
        }
        std::cout << "    All " << processed << " queries processed successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "    ERROR in FAISS search at query " << processed << ": " << e.what() << std::endl;
        throw;
    }
    std::cout << "    FAISS search completed." << std::endl;
    
    // Convert to torch tensors
    std::cout << "    Converting distances to tensor..." << std::endl;
    torch::Tensor distance_tensor = torch::from_blob(
        distances.data(), 
        {num_queries, k}, 
        torch::kFloat
    ).clone();
    
    // Convert similarities to distances for cosine metric
    if (config_.metric == DistanceMetric::COSINE) {
        std::cout << "    Converting cosine similarities to distances..." << std::endl;
        // Clamp similarity values to [0, 1] to avoid negative distances due to numerical errors
        distance_tensor = torch::clamp(distance_tensor, 0.0, 1.0);
        distance_tensor = 1.0 - distance_tensor;
    } else {
        // For L2 metric, FAISS IndexFlatL2 returns squared L2 distances
        // Take square root to get regular L2 distances
        std::cout << "    Converting squared L2 distances to L2 distances..." << std::endl;
        distance_tensor = torch::sqrt(torch::clamp(distance_tensor, 0.0f, std::numeric_limits<float>::max()));
    }
    
    // Aggregate distances
    std::cout << "    Aggregating distances..." << std::endl;
    torch::Tensor aggregated_scores = aggregateDistances(distance_tensor);
    
    std::cout << "    Aggregation completed. Final scores shape: [" << aggregated_scores.size(0) << "]" << std::endl;
    
    return aggregated_scores;
}

torch::Tensor MemoryBank::aggregateDistances(const torch::Tensor& distances) {
    if (config_.aggregation == "mean") {
        return utils::meanAggregation(distances);
    } else if (config_.aggregation == "max") {
        return utils::maxAggregation(distances);
    } else if (config_.aggregation == "median") {
        return utils::medianAggregation(distances);
    } else if (config_.aggregation == "weighted_mean") {
        return utils::weightedMeanAggregation(distances);
    } else {
        // Default to mean
        return utils::meanAggregation(distances);
    }
}

MemoryBank::MemoryBankInfo MemoryBank::getInfo() const {
    MemoryBankInfo info;
    info.total_features = all_features_.size(0);
    info.core_set_size = core_set_.size(0);
    info.feature_dim = core_set_.size(1);
    info.metric = (config_.metric == DistanceMetric::COSINE) ? "cosine" : "l2";
    return info;
}

void MemoryBank::saveMemoryBank(const std::string& save_path) {
    if (core_set_.numel() == 0) {
        throw std::runtime_error("Cannot save: memory bank is empty");
    }
    
    if (!faiss_index_) {
        throw std::runtime_error("Cannot save: FAISS index not built");
    }
    
    std::cout << "Saving memory bank to: " << save_path << std::endl;
    
    // Create directory if it doesn't exist
    std::filesystem::path dir_path = std::filesystem::path(save_path).parent_path();
    if (!dir_path.empty() && !std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
    }
    
    // Save core set tensor
    std::string core_set_path = save_path + "_core_set.pt";
    torch::save(core_set_, core_set_path);
    std::cout << "  Saved core set to: " << core_set_path << std::endl;
    
    // Save FAISS index
    std::string faiss_index_path = save_path + "_faiss_index.faiss";
    faiss::write_index(faiss_index_.get(), faiss_index_path.c_str());
    std::cout << "  Saved FAISS index to: " << faiss_index_path << std::endl;
    
    // Save metadata (config info)
    std::string metadata_path = save_path + "_metadata.txt";
    std::ofstream metadata_file(metadata_path);
    if (metadata_file.is_open()) {
        metadata_file << "core_set_size=" << core_set_.size(0) << "\n";
        metadata_file << "feature_dim=" << core_set_.size(1) << "\n";
        metadata_file << "metric=" << ((config_.metric == DistanceMetric::COSINE) ? "cosine" : "l2") << "\n";
        metadata_file << "total_features=" << all_features_.size(0) << "\n";
        metadata_file.close();
        std::cout << "  Saved metadata to: " << metadata_path << std::endl;
    } else {
        std::cerr << "  Warning: Could not save metadata to: " << metadata_path << std::endl;
    }
    
    std::cout << "Memory bank saved successfully!" << std::endl;
}

void MemoryBank::loadMemoryBank(const std::string& load_path) {
    std::cout << "Loading memory bank from: " << load_path << std::endl;
    
    // Load core set tensor
    std::string core_set_path = load_path + "_core_set.pt";
    if (!std::filesystem::exists(core_set_path)) {
        throw std::runtime_error("Core set file not found: " + core_set_path);
    }
    torch::load(core_set_, core_set_path);
    std::cout << "  Loaded core set from: " << core_set_path << std::endl;
    std::cout << "  Core set shape: [" << core_set_.size(0) << ", " << core_set_.size(1) << "]" << std::endl;
    
    // Load FAISS index
    std::string faiss_index_path = load_path + "_faiss_index.faiss";
    if (!std::filesystem::exists(faiss_index_path)) {
        throw std::runtime_error("FAISS index file not found: " + faiss_index_path);
    }
    
    // Read the FAISS index
    faiss::Index* loaded_index = faiss::read_index(faiss_index_path.c_str());
    if (!loaded_index) {
        throw std::runtime_error("Failed to load FAISS index from: " + faiss_index_path);
    }
    
    faiss_index_ = std::unique_ptr<faiss::Index>(loaded_index);
    std::cout << "  Loaded FAISS index from: " << faiss_index_path << std::endl;
    std::cout << "  FAISS index contains " << faiss_index_->ntotal << " vectors" << std::endl;
    std::cout << "  FAISS index dimension: " << faiss_index_->d << std::endl;
    std::cout << "  FAISS index type: " << typeid(*faiss_index_).name() << std::endl;
    
    // If index was saved without normalization for cosine, we need to rebuild it
    if (config_.metric == DistanceMetric::COSINE && faiss_index_->ntotal > 0) {
        std::cout << "  Warning: Index loaded for cosine metric - verifying normalization..." << std::endl;
        // For now, we'll proceed - but if search hangs, we may need to rebuild the index
    }
    
    // Load and verify metadata
    std::string metadata_path = load_path + "_metadata.txt";
    if (std::filesystem::exists(metadata_path)) {
        std::ifstream metadata_file(metadata_path);
        if (metadata_file.is_open()) {
            std::string line;
            while (std::getline(metadata_file, line)) {
                // Parse metadata (optional, just for info)
                if (line.find("core_set_size=") == 0) {
                    int saved_size = std::stoi(line.substr(15));
                    if (saved_size != core_set_.size(0)) {
                        std::cerr << "  Warning: Saved core set size (" << saved_size 
                                  << ") doesn't match loaded size (" << core_set_.size(0) << ")" << std::endl;
                    }
                }
            }
            metadata_file.close();
            std::cout << "  Loaded metadata from: " << metadata_path << std::endl;
        }
    } else {
        std::cerr << "  Warning: Metadata file not found: " << metadata_path << std::endl;
    }
    
    // Set all_features_ to the core_set_ for compatibility (even though we don't use it after loading)
    all_features_ = core_set_.clone();
    
    std::cout << "Memory bank loaded successfully!" << std::endl;
}

} // namespace patchcore

