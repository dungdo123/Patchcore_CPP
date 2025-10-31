#!/usr/bin/env python3
"""
PatchCore Resolution Handling Examples
Shows how to handle WideResNet50 layer dimensions and convert to target resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2
import numpy as np


class ResolutionAwarePatchCoreExtractor(nn.Module):
    """
    PatchCore feature extractor that handles resolution conversion properly
    """
    def __init__(self, target_resolution=32, target_dim=384):
        super().__init__()
        
        # Load WideResNet50 backbone
        self.backbone = wide_resnet50_2(pretrained=True)
        
        # Extract specific layers
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        
        # Configuration
        self.target_resolution = target_resolution
        self.target_dim = target_dim
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Conv2d(1024 + 2048, target_dim, 1),  # layer2 + layer3 channels
            nn.BatchNorm2d(target_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Extract features
        x = self.layer1(x)
        feat2 = self.layer2(x)  # [B, 512, 28, 28]
        feat3 = self.layer3(feat2)  # [B, 1024, 14, 14]
        
        print(f"Layer2 features shape: {feat2.shape}")
        print(f"Layer3 features shape: {feat3.shape}")
        
        # Method 1: Upsample both to target resolution
        feat2_upsampled = self.upsample_to_target(feat2, self.target_resolution)
        feat3_upsampled = self.upsample_to_target(feat3, self.target_resolution)
        
        print(f"Upsampled Layer2 shape: {feat2_upsampled.shape}")
        print(f"Upsampled Layer3 shape: {feat3_upsampled.shape}")
        
        # Combine features
        combined = torch.cat([feat2_upsampled, feat3_upsampled], dim=1)
        
        # Apply projection
        features = self.projection(combined)
        
        return features
    
    def upsample_to_target(self, features, target_size):
        """Upsample features to target resolution"""
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


class MultiResolutionPatchCoreExtractor(nn.Module):
    """
    Alternative approach: Use different target resolutions for different layers
    """
    def __init__(self, target_dim=384):
        super().__init__()
        
        # Load WideResNet50 backbone
        self.backbone = wide_resnet50_2(pretrained=True)
        
        # Extract specific layers
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        
        # Different target resolutions for different layers
        self.layer2_target = 32  # Upsample 28x28 to 32x32
        self.layer3_target = 32  # Upsample 14x14 to 32x32
        
        # Feature projections
        self.layer2_proj = nn.Sequential(
            nn.Conv2d(512, target_dim // 2, 1),
            nn.BatchNorm2d(target_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        self.layer3_proj = nn.Sequential(
            nn.Conv2d(1024, target_dim // 2, 1),
            nn.BatchNorm2d(target_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Conv2d(target_dim, target_dim, 1),
            nn.BatchNorm2d(target_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Extract features
        x = self.layer1(x)
        feat2 = self.layer2(x)  # [B, 512, 28, 28]
        feat3 = self.layer3(feat2)  # [B, 1024, 14, 14]
        
        # Project each layer separately
        feat2_proj = self.layer2_proj(feat2)  # [B, 192, 28, 28]
        feat3_proj = self.layer3_proj(feat3)  # [B, 192, 14, 14]
        
        # Upsample to common resolution
        feat2_upsampled = F.interpolate(
            feat2_proj, 
            size=(self.layer2_target, self.layer2_target),
            mode='bilinear', 
            align_corners=False
        )
        
        feat3_upsampled = F.interpolate(
            feat3_proj, 
            size=(self.layer3_target, self.layer3_target),
            mode='bilinear', 
            align_corners=False
        )
        
        # Combine features
        combined = torch.cat([feat2_upsampled, feat3_upsampled], dim=1)
        
        # Final projection
        features = self.final_proj(combined)
        
        return features


class AdaptiveResolutionPatchCoreExtractor(nn.Module):
    """
    Adaptive approach: Choose resolution based on layer characteristics
    """
    def __init__(self, target_dim=384):
        super().__init__()
        
        # Load WideResNet50 backbone
        self.backbone = wide_resnet50_2(pretrained=True)
        
        # Extract specific layers
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        
        # Adaptive resolution strategy
        self.use_layer2_native = True   # Keep layer2 at 28x28
        self.use_layer3_upsampled = True  # Upsample layer3 to 28x28
        
        # Feature projections
        if self.use_layer2_native:
            self.layer2_proj = nn.Sequential(
                nn.Conv2d(512, target_dim // 2, 1),
                nn.BatchNorm2d(target_dim // 2),
                nn.ReLU(inplace=True)
            )
        
        if self.use_layer3_upsampled:
            self.layer3_proj = nn.Sequential(
                nn.Conv2d(1024, target_dim // 2, 1),
                nn.BatchNorm2d(target_dim // 2),
                nn.ReLU(inplace=True)
            )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Conv2d(target_dim, target_dim, 1),
            nn.BatchNorm2d(target_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Extract features
        x = self.layer1(x)
        feat2 = self.layer2(x)  # [B, 512, 28, 28]
        feat3 = self.layer3(feat2)  # [B, 1024, 14, 14]
        
        # Process layer2 (keep native resolution)
        if self.use_layer2_native:
            feat2_proj = self.layer2_proj(feat2)  # [B, 192, 28, 28]
        else:
            feat2_proj = feat2
        
        # Process layer3 (upsample to match layer2)
        if self.use_layer3_upsampled:
            feat3_proj = self.layer3_proj(feat3)  # [B, 192, 14, 14]
            # Upsample to match layer2 resolution
            feat3_upsampled = F.interpolate(
                feat3_proj, 
                size=feat2_proj.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
        else:
            feat3_upsampled = feat3
        
        # Combine features
        combined = torch.cat([feat2_proj, feat3_upsampled], dim=1)
        
        # Final projection
        features = self.final_proj(combined)
        
        return features


def demonstrate_resolution_handling():
    """Demonstrate different resolution handling strategies"""
    
    print("=== PatchCore Resolution Handling Demo ===\n")
    
    # Create example input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 256, 256)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Target resolution: 32x32\n")
    
    # Strategy 1: Simple upsampling
    print("1. Simple Upsampling Strategy:")
    print("-" * 40)
    extractor1 = ResolutionAwarePatchCoreExtractor(target_resolution=32)
    extractor1.eval()
    
    with torch.no_grad():
        output1 = extractor1(input_tensor)
        print(f"Output shape: {output1.shape}\n")
    
    # Strategy 2: Multi-resolution
    print("2. Multi-Resolution Strategy:")
    print("-" * 40)
    extractor2 = MultiResolutionPatchCoreExtractor()
    extractor2.eval()
    
    with torch.no_grad():
        output2 = extractor2(input_tensor)
        print(f"Output shape: {output2.shape}\n")
    
    # Strategy 3: Adaptive resolution
    print("3. Adaptive Resolution Strategy:")
    print("-" * 40)
    extractor3 = AdaptiveResolutionPatchCoreExtractor()
    extractor3.eval()
    
    with torch.no_grad():
        output3 = extractor3(input_tensor)
        print(f"Output shape: {output3.shape}\n")
    
    # Compare memory usage
    print("Memory Usage Comparison:")
    print("-" * 40)
    print(f"Strategy 1 (32x32): {output1.numel() * 4 / 1024 / 1024:.2f} MB")
    print(f"Strategy 2 (32x32): {output2.numel() * 4 / 1024 / 1024:.2f} MB")
    print(f"Strategy 3 (28x28): {output3.numel() * 4 / 1024 / 1024:.2f} MB")


def create_libtorch_model_with_resolution_handling():
    """Create LibTorch model with proper resolution handling"""
    
    print("\n=== Creating LibTorch Model with Resolution Handling ===\n")
    
    # Create extractor
    extractor = ResolutionAwarePatchCoreExtractor(target_resolution=32)
    extractor.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, 256, 256)
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = extractor(example_input)
        print(f"Output shape: {output.shape}")
    
    # Trace the model
    print("Tracing model...")
    traced_model = torch.jit.trace(extractor, example_input)
    
    # Save traced model
    output_path = "models/patchcore_resolution_aware.pt"
    traced_model.save(output_path)
    print(f"LibTorch model saved to: {output_path}")
    
    # Verify the saved model
    print("Verifying saved model...")
    loaded_model = torch.jit.load(output_path)
    with torch.no_grad():
        test_output = loaded_model(example_input)
        print(f"Loaded model output shape: {test_output.shape}")
    
    return output_path


if __name__ == "__main__":
    # Demonstrate resolution handling
    demonstrate_resolution_handling()
    
    # Create LibTorch model
    model_path = create_libtorch_model_with_resolution_handling()
    
    print(f"\n✅ Resolution handling demo completed!")
    print(f"📁 Model saved to: {model_path}")
    print(f"🔧 Use this model with the C++ implementation")
