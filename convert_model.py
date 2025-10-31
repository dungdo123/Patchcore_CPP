#!/usr/bin/env python3
"""
Convert PyTorch PatchCore model to LibTorch format for C++ inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
import argparse
import os


class PatchCoreFeatureExtractor(nn.Module):
    """
    PatchCore feature extractor matching official implementation
    Extracts features from WideResNet50 layer2 and layer3, then projects to target dimension
    """
    def __init__(self, target_dim=384):
        super().__init__()
        
        # Load WideResNet50 backbone with pretrained weights
        try:
            # Try new API first (torchvision >= 0.13)
            self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        except (AttributeError, TypeError):
            # Fallback to old API for older torchvision versions
            self.backbone = wide_resnet50_2(pretrained=True)
        
        # Extract specific layers
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        
        # Feature projection layer - will be created dynamically based on actual feature dimensions
        # The actual channel dimensions depend on the WideResNet variant
        self.target_dim = target_dim
        self.projection = None
        
    def forward(self, x):
        """
        Forward pass following official PatchCore implementation:
        1. Extract features from layer2 and layer3
        2. Upsample layer3 to match layer2 spatial resolution
        3. Concatenate features
        4. Project to target dimension
        """
        # Initial layers
        print(f"Input size: {x.size(2)}x{x.size(3)}")
        x = self.conv1(x)
        print(f"After conv1: {x.size(2)}x{x.size(3)}")
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(f"After maxpool: {x.size(2)}x{x.size(3)}")
        
        # Extract features
        x = self.layer1(x)
        print(f"After layer1: {x.size(2)}x{x.size(3)}")
        feat2 = self.layer2(x)
        print(f"After layer2: {feat2.size(2)}x{feat2.size(3)} (channels: {feat2.size(1)})")
        feat3 = self.layer3(feat2)
        print(f"After layer3: {feat3.size(2)}x{feat3.size(3)} (channels: {feat3.size(1)})")
        
        # Upsample feat3 to match feat2's spatial dimensions
        print(f"Upsampling layer3 from {feat3.size(2)}x{feat3.size(3)} to match layer2: {feat2.size(2)}x{feat2.size(3)}")
        feat3_upsampled = F.interpolate(
            feat3, 
            size=(feat2.size(2), feat2.size(3)), 
            mode='bilinear', 
            align_corners=False
        )
        print(f"After upsampling layer3: {feat3_upsampled.size(2)}x{feat3_upsampled.size(3)}")
        
        # Combine features from layer2 and layer3
        combined = torch.cat([feat2, feat3_upsampled], dim=1)
        print(f"After concatenation: {combined.size(2)}x{combined.size(3)} (channels: {combined.size(1)})")
        
        # Create projection layer dynamically if it doesn't exist (based on actual dimensions)
        if self.projection is None:
            input_channels = combined.size(1)
            spatial_size = combined.size(2)  # Height and width should be the same
            print(f"Detected feature channels: layer2={feat2.size(1)}, layer3={feat3.size(1)}, combined={input_channels}")
            print(f"Spatial size after concatenation: {spatial_size}x{spatial_size}")
            self.projection = nn.Conv2d(input_channels, self.target_dim, kernel_size=1, stride=1, padding=0, bias=True)
            # Initialize projection weights properly (Kaiming uniform)
            nn.init.kaiming_uniform_(self.projection.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.projection.bias)
            print(f"Created projection layer: {input_channels} -> {self.target_dim} channels")
        
        # Apply projection to target dimension (1x1 conv preserves spatial dimensions)
        features = self.projection(combined)
        print(f"Final model output: {features.size(2)}x{features.size(3)} (channels: {features.size(1)})")
        print("=" * 50)
        
        return features


def convert_model_to_libtorch(model_path, output_path, target_dim=384):
    """
    Convert PyTorch model to LibTorch format
    """
    print(f"Converting model to LibTorch format...")
    print(f"Input model: {model_path}")
    print(f"Output path: {output_path}")
    
    # Create feature extractor
    extractor = PatchCoreFeatureExtractor(target_dim=target_dim)
    extractor.eval()
    
    # Load weights if model path exists
    if os.path.exists(model_path):
        print(f"Loading weights from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            extractor.load_state_dict(checkpoint['model_state_dict'])
        else:
            extractor.load_state_dict(checkpoint)
    else:
        print(f"Model path {model_path} not found, using pretrained weights")
    
    # Create example input
    example_input = torch.randn(1, 3, 256, 256)
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = extractor(example_input)
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: [1, {target_dim}, H, W]")
    
    # Use scripting for better compatibility (supports dynamic control flow)
    print("Scripting model...")
    try:
        scripted_model = torch.jit.script(extractor)
        print("Model scripted successfully")
    except Exception as e:
        print(f"Scripting failed: {e}")
        print("Falling back to tracing...")
        # Fallback to tracing if scripting fails
        scripted_model = torch.jit.trace(extractor, example_input)
    
    # Save scripted/traced model
    scripted_model.save(output_path)
    print(f"LibTorch model saved to: {output_path}")
    
    # Verify the saved model
    print("Verifying saved model...")
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()
    with torch.no_grad():
        test_output = loaded_model(example_input)
        print(f"Loaded model output shape: {test_output.shape}")
        if test_output.shape[1] != target_dim:
            print(f"WARNING: Output dimension {test_output.shape[1]} doesn't match target {target_dim}")
        else:
            print("Model verification successful!")
    
    return output_path


def create_sample_model(output_path, target_dim=384):
    """
    Create a sample PatchCore model for testing
    """
    print(f"Creating sample PatchCore model...")
    
    # Create feature extractor
    extractor = PatchCoreFeatureExtractor(target_dim=target_dim)
    extractor.eval()
    
    # Create example input
    example_input = torch.randn(1, 3, 256, 256)
    
    # Test forward pass
    print("Testing forward pass...")
    with torch.no_grad():
        output = extractor(example_input)
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: [1, {target_dim}, H, W]")
    
    # Use scripting for better compatibility
    print("Scripting model...")
    try:
        scripted_model = torch.jit.script(extractor)
        print("Model scripted successfully")
    except Exception as e:
        print(f"Scripting failed: {e}")
        print("Falling back to tracing...")
        # Fallback to tracing if scripting fails
        scripted_model = torch.jit.trace(extractor, example_input)
    
    # Save scripted/traced model
    scripted_model.save(output_path)
    print(f"Sample LibTorch model saved to: {output_path}")
    
    # Verify the saved model
    print("Verifying saved model...")
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()
    with torch.no_grad():
        test_output = loaded_model(example_input)
        print(f"Loaded model output shape: {test_output.shape}")
        if test_output.shape[1] != target_dim:
            print(f"WARNING: Output dimension {test_output.shape[1]} doesn't match target {target_dim}")
        else:
            print("Model verification successful!")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch PatchCore model to LibTorch format')
    parser.add_argument('--input', type=str, default=None,
                       help='Input PyTorch model path')
    parser.add_argument('--output', type=str, default='models/patchcore_extractor_v2.pt',
                       help='Output LibTorch model path')
    parser.add_argument('--target-dim', type=int, default=384,
                       help='Target feature dimension')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample model for testing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.create_sample:
        # Create sample model
        create_sample_model(args.output, args.target_dim)
    else:
        # Convert existing model
        if args.input is None:
            print("Error: --input path is required when not creating sample model")
            return 1
        
        convert_model_to_libtorch(args.input, args.output, args.target_dim)
    
    print("Model conversion completed successfully!")
    return 0


if __name__ == '__main__':
    exit(main())

