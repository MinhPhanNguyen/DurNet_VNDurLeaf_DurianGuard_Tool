#!/usr/bin/env python3
"""
Script để kiểm tra cấu trúc của model weights
"""

import torch

def inspect_model_weights(model_path):
    """Kiểm tra cấu trúc của model weights"""
    try:
        print(f"Đang kiểm tra model: {model_path}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(checkpoint, dict):
            print("Model format: Dictionary")
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print("Found 'model_state_dict' key")
            else:
                state_dict = checkpoint
                print("Using checkpoint as state_dict")
        else:
            state_dict = checkpoint
            print("Model format: Direct state_dict")
        
        print(f"\nTotal parameters: {len(state_dict)}")
        print("\nLayer structure:")
        
        # Group layers by prefix
        layer_groups = {}
        for key in state_dict.keys():
            parts = key.split('.')
            prefix = parts[0] if len(parts) > 1 else key
            
            if prefix not in layer_groups:
                layer_groups[prefix] = []
            layer_groups[prefix].append(key)
        
        for prefix, layers in layer_groups.items():
            print(f"\n{prefix}: {len(layers)} parameters")
            for layer in layers[:3]:  # Show first 3
                shape = state_dict[layer].shape if hasattr(state_dict[layer], 'shape') else 'scalar'
                print(f"  {layer}: {shape}")
            if len(layers) > 3:
                print(f"  ... and {len(layers) - 3} more")
                
        # Try to determine model architecture
        print("\n" + "="*50)
        print("ARCHITECTURE ANALYSIS:")
        
        if 'conv1.weight' in state_dict:
            print("✅ Found conv1 - likely custom CNN")
        if 'backbone.features' in str(list(state_dict.keys())):
            print("✅ Found backbone.features - likely using pretrained backbone")
        if 'fc.weight' in state_dict:
            print("✅ Found fc - final classification layer")
        if 'block1' in str(list(state_dict.keys())):
            print("✅ Found block structure - likely ResNet-style")
        if 'middle_blocks' in str(list(state_dict.keys())):
            print("✅ Found middle_blocks - custom architecture")
            
        # Analyze final layer
        if 'fc.weight' in state_dict:
            fc_shape = state_dict['fc.weight'].shape
            print(f"📊 Final layer (fc): {fc_shape}")
            print(f"   Number of classes: {fc_shape[0]}")
            print(f"   Input features: {fc_shape[1]}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == '__main__':
    model_path = 'durnet.pth'
    inspect_model_weights(model_path)