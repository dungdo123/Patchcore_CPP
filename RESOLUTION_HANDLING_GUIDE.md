# PatchCore Resolution Handling Guide

## 🎯 **The Resolution Challenge**

### **WideResNet50 Layer Dimensions:**
```
Input: 256×256
├── Conv1 + MaxPool: 64×64
├── Layer1: 64×64
├── Layer2: 28×28  ← Target: 32×32
└── Layer3: 14×14  ← Target: 32×32
```

### **The Problem:**
- **Layer2**: 28×28 → Need 32×32 (upsample by 1.14×)
- **Layer3**: 14×14 → Need 32×32 (upsample by 2.29×)
- **Challenge**: How to handle different upsampling ratios?

## 🔧 **Solution Strategies**

### **Strategy 1: Simple Upsampling (Recommended)**

```python
def upsample_to_target(features, target_size):
    """Upsample features to target resolution using bilinear interpolation"""
    current_size = features.shape[2]  # Assuming square features
    
    if current_size == target_size:
        return features
    
    # Use bilinear interpolation
    upsampled = F.interpolate(
        features,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    )
    
    return upsampled

# Usage:
feat2_upsampled = upsample_to_target(feat2, 32)  # 28×28 → 32×32
feat3_upsampled = upsample_to_target(feat3, 32)  # 14×14 → 32×32
```

**Pros:**
- ✅ Simple and straightforward
- ✅ Maintains spatial relationships
- ✅ Works well with bilinear interpolation
- ✅ Consistent resolution across layers

**Cons:**
- ⚠️ Slight information loss due to upsampling
- ⚠️ Layer3 upsampling is more aggressive (2.29×)

### **Strategy 2: Multi-Resolution Approach**

```python
class MultiResolutionExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Different target resolutions
        self.layer2_target = 32  # 28×28 → 32×32
        self.layer3_target = 32  # 14×14 → 32×32
        
        # Separate projections
        self.layer2_proj = nn.Conv2d(512, 192, 1)
        self.layer3_proj = nn.Conv2d(1024, 192, 1)
    
    def forward(self, x):
        feat2 = self.layer2(x)  # [B, 512, 28, 28]
        feat3 = self.layer3(feat2)  # [B, 1024, 14, 14]
        
        # Project first, then upsample
        feat2_proj = self.layer2_proj(feat2)  # [B, 192, 28, 28]
        feat3_proj = self.layer3_proj(feat3)  # [B, 192, 14, 14]
        
        # Upsample to common resolution
        feat2_upsampled = F.interpolate(feat2_proj, size=(32, 32))
        feat3_upsampled = F.interpolate(feat3_proj, size=(32, 32))
        
        # Combine
        combined = torch.cat([feat2_upsampled, feat3_upsampled], dim=1)
        return combined
```

**Pros:**
- ✅ Better feature preservation
- ✅ Separate projections for each layer
- ✅ More control over feature combination

**Cons:**
- ⚠️ More complex implementation
- ⚠️ Higher memory usage

### **Strategy 3: Adaptive Resolution**

```python
class AdaptiveResolutionExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Keep layer2 at native resolution (28×28)
        # Upsample layer3 to match layer2 (28×28)
        self.target_resolution = 28  # Use layer2's native resolution
    
    def forward(self, x):
        feat2 = self.layer2(x)  # [B, 512, 28, 28]
        feat3 = self.layer3(feat2)  # [B, 1024, 14, 14]
        
        # Keep layer2 as is
        feat2_processed = self.layer2_proj(feat2)  # [B, 192, 28, 28]
        
        # Upsample layer3 to match layer2
        feat3_upsampled = F.interpolate(
            self.layer3_proj(feat3), 
            size=feat2_processed.shape[2:],  # Match layer2 size
            mode='bilinear'
        )
        
        # Combine
        combined = torch.cat([feat2_processed, feat3_upsampled], dim=1)
        return combined
```

**Pros:**
- ✅ Minimal upsampling (only layer3)
- ✅ Preserves layer2's native resolution
- ✅ More efficient memory usage

**Cons:**
- ⚠️ Non-standard resolution (28×28 instead of 32×32)
- ⚠️ May not match your target requirements

## 🚀 **Implementation in C++**

### **Updated Feature Extractor:**

```cpp
torch::Tensor FeatureExtractor::upsampleToTargetResolution(torch::Tensor features, int target_size) {
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
```

### **Usage in Feature Extraction:**

```cpp
torch::Tensor FeatureExtractor::extractFeaturesBatch(const std::vector<cv::Mat>& images) {
    // ... existing code ...
    
    // Extract features from specified layers
    std::vector<torch::Tensor> layer_features;
    
    for (const auto& layer_name : config_.layers_to_extract) {
        torch::Tensor layer_feat = extractLayerFeatures(output, layer_name);
        
        // Upsample to target resolution (32x32)
        layer_feat = upsampleToTargetResolution(layer_feat, 32);
        
        layer_features.push_back(layer_feat);
    }
    
    // Combine features from different layers
    torch::Tensor combined_features = torch::cat(layer_features, 1);
    
    // ... rest of the code ...
}
```

## 📊 **Resolution Comparison**

| Strategy | Layer2 | Layer3 | Final Resolution | Memory Usage | Quality |
|----------|--------|--------|------------------|--------------|---------|
| Simple Upsampling | 28→32 | 14→32 | 32×32 | High | Good |
| Multi-Resolution | 28→32 | 14→32 | 32×32 | Very High | Very Good |
| Adaptive | 28→28 | 14→28 | 28×28 | Medium | Good |

## 🎯 **Recommendations**

### **For Production Use:**
1. **Use Strategy 1 (Simple Upsampling)** - Most practical
2. **Target Resolution: 32×32** - Matches your requirements
3. **Bilinear Interpolation** - Good balance of quality and speed

### **For Research/Experimentation:**
1. **Try Strategy 2 (Multi-Resolution)** - Better feature preservation
2. **Compare different target resolutions** - 28×28, 32×32, 64×64
3. **Experiment with interpolation methods** - Bilinear, nearest, bicubic

### **For Memory-Constrained Environments:**
1. **Use Strategy 3 (Adaptive)** - Lower memory usage
2. **Consider 28×28 resolution** - Native layer2 resolution
3. **Optimize feature dimensions** - Reduce target_dim

## 🔧 **Practical Implementation**

### **Step 1: Create Python Model**
```bash
python3 resolution_handling_demo.py
```

### **Step 2: Convert to LibTorch**
```bash
python3 convert_model.py --create-sample --output models/patchcore_32x32.pt
```

### **Step 3: Use in C++**
```bash
./patchcore_detector \
    models/patchcore_32x32.pt \
    normal_images/ \
    test_images/ \
    --output-dir results/
```

## 📝 **Key Takeaways**

1. **Resolution Mismatch**: WideResNet50 layers have different spatial dimensions
2. **Upsampling Solution**: Use bilinear interpolation to reach target resolution
3. **Strategy Choice**: Simple upsampling is recommended for most cases
4. **C++ Implementation**: Updated to handle resolution conversion automatically
5. **Model Preparation**: Use Python script to create properly configured models

The C++ implementation now automatically handles the resolution conversion, so you can focus on using the detector without worrying about the underlying resolution details!

