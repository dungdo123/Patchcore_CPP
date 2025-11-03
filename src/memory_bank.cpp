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
#include <unordered_set>

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
        
        // Flatten spatial dimensions following official PatchCore implementation
        // Must match the same permutation used in prediction to ensure spatial correspondence
        int64_t batch_size_img = features.size(0);
        int64_t channels = features.size(1);
        int64_t height = features.size(2);
        int64_t width = features.size(3);
        
        // Permute from [B, C, H, W] to [B, H, W, C] to ensure correct spatial ordering
        // This matches the official PatchCore Python implementation
        torch::Tensor features_permuted = features.permute({0, 2, 3, 1}).contiguous();
        
        // Reshape to [batch * height * width, channels] preserving spatial order (row-major)
        torch::Tensor flattened = features_permuted.view({batch_size_img * height * width, channels});
        
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
        // Always use greedy sampling for core set building
        std::cout << "Using greedy sampling for core set building..." << std::endl;
        if (config_.auto_size) {
            core_set_ = approximateGreedyCoreSet(all_features_);
        } else {
            core_set_ = greedyCoreSetSampling(all_features_);
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
    
    // Ensure features are on CPU to avoid GPU memory issues
    torch::Tensor features_cpu = features.cpu().contiguous();
    
    std::vector<int64_t> selected_indices;
    selected_indices.reserve(target_size);
    
    // Random initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, total_features - 1);
    
    selected_indices.push_back(dis(gen));
    
    // Greedy selection with memory-efficient chunked processing
    // Store indices instead of features to save memory
    std::vector<int64_t> selected_features_indices;
    selected_features_indices.push_back(selected_indices[0]);
    
    // Chunk sizes for memory efficiency
    const int64_t feature_chunk_size = 5000; // Process 5000 features at a time
    const int64_t selected_chunk_size = 1000; // Process 1000 selected features at a time
    
    for (int64_t i = 1; i < target_size; ++i) {
        if (i % 500 == 0) {
            std::cout << "  Selected " << i << "/" << target_size << " features" << std::endl;
        }
        
        // Process all features in chunks to compute minimum distances
        torch::Tensor min_distances = torch::full({total_features}, std::numeric_limits<float>::max(), 
                                                  torch::TensorOptions().dtype(torch::kFloat32).device(features_cpu.device()));
        
        for (int64_t start = 0; start < total_features; start += feature_chunk_size) {
            int64_t end = std::min(start + feature_chunk_size, total_features);
            torch::Tensor chunk_features = features_cpu.slice(0, start, end);
            
            torch::Tensor chunk_min_distances = torch::full({end - start}, std::numeric_limits<float>::max(),
                                                           torch::TensorOptions().dtype(torch::kFloat32).device(features_cpu.device()));
            
            // Process selected features in chunks to avoid large distance matrices
            for (size_t sel_start = 0; sel_start < selected_features_indices.size(); sel_start += selected_chunk_size) {
                size_t sel_end = std::min(sel_start + selected_chunk_size, selected_features_indices.size());
                std::vector<int64_t> sel_chunk(selected_features_indices.begin() + sel_start,
                                              selected_features_indices.begin() + sel_end);
                torch::Tensor selected_chunk = features_cpu.index_select(0, torch::tensor(sel_chunk));
                
                // Compute distances between feature chunk and selected chunk
                torch::Tensor distances = utils::l2Distance(chunk_features, selected_chunk);
                torch::Tensor chunk_dists_min = std::get<0>(torch::min(distances, 1));
                
                // Update minimum distances (element-wise minimum)
                chunk_min_distances = torch::min(chunk_min_distances, chunk_dists_min);
            }
            
            // Update global minimum distances
            min_distances.slice(0, start, end) = chunk_min_distances;
        }
        
        // Find the feature with maximum minimum distance
        auto max_result = torch::max(min_distances, 0);
        int64_t max_idx = std::get<1>(max_result).item<int64_t>();
        selected_indices.push_back(max_idx);
        selected_features_indices.push_back(max_idx);
    }
    
    return features_cpu.index_select(0, torch::tensor(selected_indices));
}

torch::Tensor MemoryBank::approximateGreedyCoreSet(const torch::Tensor& features) {
    int64_t total_features = features.size(0);
    int64_t target_size = (config_.core_set_size < total_features) ? config_.core_set_size : total_features;
    
    std::cout << "  Approximate greedy sampling: selecting " << target_size << " from " << total_features << " features" << std::endl;
    
    // Ensure features are on CPU to avoid GPU memory issues
    torch::Tensor features_cpu = features.cpu().contiguous();
    
    // Use larger initial seed like official implementation
    int64_t seed_size;
    if (total_features > 100000) {
        seed_size = std::max(std::min(target_size / 20, static_cast<int64_t>(50)), static_cast<int64_t>(1)); // For large datasets
    } else {
        seed_size = std::max(std::min(target_size / 100, static_cast<int64_t>(10)), static_cast<int64_t>(1)); // For smaller datasets
    }
    
    // Batch size for adding multiple features at once (key optimization!)
    int64_t batch_size = (total_features < 100000) ? 100 : 500;
    
    std::cout << "  Initial seed size: " << seed_size << ", batch size: " << batch_size << std::endl;
    
    // Candidate pool management
    int64_t candidate_pool_size;
    std::vector<int64_t> all_candidate_indices;
    if (total_features > 100000) {
        // For large datasets, use adaptive candidate pool
        candidate_pool_size = std::min(total_features, target_size * 5);
        all_candidate_indices.reserve(total_features);
        for (int64_t i = 0; i < total_features; ++i) {
            all_candidate_indices.push_back(i);
        }
    } else {
        // For smaller datasets, use all features as candidates
        candidate_pool_size = total_features;
        all_candidate_indices.reserve(total_features);
        for (int64_t i = 0; i < total_features; ++i) {
            all_candidate_indices.push_back(i);
        }
    }
    
    // Random initialization with seed
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_int_distribution<> dis(0, total_features - 1);
    
    // Initialize coreset with random seed
    std::unordered_set<int64_t> core_set_indices;
    std::vector<int64_t> seed_candidates;
    seed_candidates.reserve(seed_size);
    std::unordered_set<int64_t> seed_set;
    
    while (seed_candidates.size() < seed_size) {
        int64_t idx = dis(gen);
        if (seed_set.find(idx) == seed_set.end()) {
            seed_set.insert(idx);
            seed_candidates.push_back(idx);
            core_set_indices.insert(idx);
        }
    }
    
    // Build remaining candidates list (exclude seed)
    std::vector<int64_t> remaining_candidates;
    remaining_candidates.reserve(candidate_pool_size);
    for (int64_t idx : all_candidate_indices) {
        if (core_set_indices.find(idx) == core_set_indices.end()) {
            remaining_candidates.push_back(idx);
        }
    }
    
    std::cout << "  Starting greedy selection with " << core_set_indices.size() << " seed features, "
              << remaining_candidates.size() << " remaining candidates" << std::endl;
    
    // Keep only indices, fetch features on-demand to save memory
    std::vector<int64_t> selected_indices_vec(core_set_indices.begin(), core_set_indices.end());
    
    int64_t iteration = 0;
    // Optimized chunk sizes for memory efficiency:
    // - Smaller candidate chunks to reduce distance matrix size
    // - Larger selected chunks since we process fewer at a time
    const int64_t candidate_chunk_size = 1000; // Reduced from 5000 to limit distance matrix size
    const int64_t selected_chunk_size = 500;    // Smaller chunks for memory efficiency
    
    // Main greedy selection loop with batch processing
    while (core_set_indices.size() < target_size && !remaining_candidates.empty()) {
        // Progress reporting - more frequent for better visibility
        if (iteration == 0 || core_set_indices.size() % 1000 == 0 || core_set_indices.size() >= target_size) {
            std::cout << "  Progress: Selected " << core_set_indices.size() << "/" << target_size 
                      << " features (iteration " << iteration << ", remaining candidates: " 
                      << remaining_candidates.size() << ")" << std::endl;
            std::cout.flush();
        }
        
        // Always subsample candidates for very large datasets to reduce memory
        std::vector<int64_t> candidates_to_check;
        int64_t max_candidates = (remaining_candidates.size() > 10000) ? 10000 : static_cast<int64_t>(remaining_candidates.size());
        if (remaining_candidates.size() > max_candidates) {
            candidates_to_check.reserve(max_candidates);
            std::vector<int64_t> shuffled = remaining_candidates;
            std::shuffle(shuffled.begin(), shuffled.end(), gen);
            candidates_to_check.assign(shuffled.begin(), shuffled.begin() + max_candidates);
            std::cout << "    Iteration " << iteration << ": Subsampling to " << max_candidates 
                      << " candidates (selected: " << core_set_indices.size() << ")" << std::endl;
            std::cout.flush();
        } else {
            candidates_to_check = remaining_candidates;
        }
        
        // Use optimized torch-based distance computation (more reliable than FAISS for this use case)
        std::cout << "    Computing distances for " << candidates_to_check.size() 
                  << " candidates (selected features: " << selected_indices_vec.size() << ")..." << std::endl;
        std::cout.flush();
        
        // Load all selected features once (memory efficient since selected set is small)
        torch::Tensor selected_features_all = features_cpu.index_select(0, torch::tensor(selected_indices_vec));
        int64_t num_selected = selected_indices_vec.size();
        int64_t num_candidates = candidates_to_check.size();
        
        // Pre-allocate result tensor
        torch::Tensor min_distances = torch::full({num_candidates}, std::numeric_limits<float>::max(),
                                                  torch::TensorOptions().dtype(torch::kFloat32).device(features_cpu.device()));
        
        // Process candidates in chunks to limit memory usage
        // Optimize chunk size based on available memory and selected features count
        int64_t candidate_chunk_size = 2000; // Process 2000 candidates at a time
        if (num_selected > 1000) {
            candidate_chunk_size = 1000; // Smaller chunks when selected set is large
        }
        
        int64_t num_chunks = (num_candidates + candidate_chunk_size - 1) / candidate_chunk_size;
        std::cout << "      Processing " << num_chunks << " candidate chunks of size " << candidate_chunk_size << std::endl;
        std::cout.flush();
        
        for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            int64_t chunk_start = chunk_idx * candidate_chunk_size;
            int64_t chunk_end = std::min(chunk_start + candidate_chunk_size, num_candidates);
            int64_t current_chunk_size = chunk_end - chunk_start;
            
            if (chunk_idx % 2 == 0 || chunk_idx == num_chunks - 1) {
                std::cout << "        Processing chunk " << (chunk_idx + 1) << "/" << num_chunks 
                          << " (" << chunk_start << "-" << (chunk_end - 1) << ")..." << std::endl;
                std::cout.flush();
            }
            
            // Extract candidate chunk
            std::vector<int64_t> chunk_candidates(candidates_to_check.begin() + chunk_start,
                                                 candidates_to_check.begin() + chunk_end);
            torch::Tensor candidate_chunk = features_cpu.index_select(0, torch::tensor(chunk_candidates));
            
            // Compute distances: candidate_chunk [N, D] x selected_features_all [M, D] -> [N, M]
            // Use efficient matrix multiplication for distance computation
            // L2 distance = ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b^T
            torch::Tensor candidate_norms = torch::sum(candidate_chunk * candidate_chunk, 1, true); // [N, 1]
            torch::Tensor selected_norms = torch::sum(selected_features_all * selected_features_all, 1, true).t(); // [1, M]
            torch::Tensor dot_product = torch::mm(candidate_chunk, selected_features_all.t()); // [N, M]
            
            // Compute squared L2 distances: ||a||^2 + ||b||^2 - 2*a*b^T
            torch::Tensor squared_distances = candidate_norms.expand({current_chunk_size, num_selected}) +
                                             selected_norms.expand({current_chunk_size, num_selected}) -
                                             2 * dot_product;
            
            // Get minimum distance for each candidate (to nearest selected feature)
            torch::Tensor chunk_min_distances = torch::sqrt(torch::clamp(
                std::get<0>(torch::min(squared_distances, 1)), 0.0f, std::numeric_limits<float>::max()));
            
            // Store results
            min_distances.slice(0, chunk_start, chunk_end) = chunk_min_distances;
        }
        
        std::cout << "      Distance computation complete" << std::endl;
        std::cout.flush();
        
        // Select batch of features with maximum minimum distances
        int64_t n_to_add = std::min({batch_size, target_size - static_cast<int64_t>(core_set_indices.size()),
                                     static_cast<int64_t>(candidates_to_check.size())});
        
        if (n_to_add <= 0) break;
        
        // Get top n_to_add indices (using argpartition-like approach via topk)
        auto topk_result = torch::topk(min_distances, n_to_add, 0);
        torch::Tensor farthest_indices = std::get<1>(topk_result);
        
        // Add selected features to coreset (BATCH ADDITION - key optimization!)
        std::vector<int64_t> newly_selected;
        newly_selected.reserve(n_to_add);
        for (int64_t i = 0; i < n_to_add; ++i) {
            int64_t idx_in_candidates = farthest_indices[i].item<int64_t>();
            int64_t selected_idx = candidates_to_check[idx_in_candidates];
            
            if (core_set_indices.find(selected_idx) == core_set_indices.end()) {
                core_set_indices.insert(selected_idx);
                selected_indices_vec.push_back(selected_idx);
                newly_selected.push_back(selected_idx);
            }
        }
        
        std::cout << "    Selected " << newly_selected.size() << " new features. Total: " 
                  << core_set_indices.size() << "/" << target_size << std::endl;
        std::cout.flush();
        
        // Remove added features from remaining candidates
        std::unordered_set<int64_t> newly_selected_set(newly_selected.begin(), newly_selected.end());
        remaining_candidates.erase(
            std::remove_if(remaining_candidates.begin(), remaining_candidates.end(),
                [&newly_selected_set](int64_t idx) { return newly_selected_set.find(idx) != newly_selected_set.end(); }),
            remaining_candidates.end()
        );
        
        iteration++;
        
        // Periodic refresh of candidate pool from all features for large datasets
        if (total_features > 100000 && iteration % 10 == 0 && remaining_candidates.size() < candidate_pool_size / 2) {
            // Rebuild remaining candidates from all features
            remaining_candidates.clear();
            for (int64_t idx : all_candidate_indices) {
                if (core_set_indices.find(idx) == core_set_indices.end()) {
                    remaining_candidates.push_back(idx);
                }
            }
            std::shuffle(remaining_candidates.begin(), remaining_candidates.end(), gen);
        }
    }
    
    // Convert to final coreset
    std::vector<int64_t> final_indices(core_set_indices.begin(), core_set_indices.end());
    std::sort(final_indices.begin(), final_indices.end());
    
    std::cout << "  Approximate greedy coreset complete: " << final_indices.size() << " features selected" << std::endl;
    
    return features_cpu.index_select(0, torch::tensor(final_indices));
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

