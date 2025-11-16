"""
PyTorch Model Inspection Script

This script loads and inspects the optimizer.pt model to determine:
- Model architecture
- Input shape requirements
- Output classes and their order
- Model parameters and size
"""

import torch
import os
import sys

def inspect_model(model_path: str):
    """Inspect the PyTorch model and print detailed information."""
    
    print("=" * 80)
    print("PyTorch Model Inspection")
    print("=" * 80)
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Looking for: {os.path.abspath(model_path)}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"   File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    print()
    
    try:
        # Load the model
        print("Loading model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = torch.load(model_path, map_location=device)
        print("✅ Model loaded successfully")
        print()
        
        # Check what was loaded
        print("Model Type:")
        print(f"  Type: {type(model)}")
        print()
        
        # If it's a state dict, we can't proceed without architecture
        if isinstance(model, dict):
            print("⚠️  Loaded object is a state dictionary (weights only)")
            print("   Keys in state dict:")
            for key in list(model.keys())[:10]:
                print(f"     - {key}")
            if len(model.keys()) > 10:
                print(f"     ... and {len(model.keys()) - 10} more")
            print()
            print("❌ Cannot determine architecture from state dict alone")
            print("   Need the model architecture definition to load weights")
            return False
        
        # If it's a complete model
        print("Model Architecture:")
        print(model)
        print()
        
        # Try to get model summary
        print("Model Structure:")
        if hasattr(model, 'modules'):
            for name, module in model.named_modules():
                if name:  # Skip the root module
                    print(f"  {name}: {module.__class__.__name__}")
        print()
        
        # Count parameters
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Parameters:")
            print(f"  Total: {total_params:,}")
            print(f"  Trainable: {trainable_params:,}")
            print()
        
        # Try to determine input shape
        print("Attempting to determine input shape...")
        model.eval()
        
        # Try common input shapes
        test_shapes = [
            (1, 3, 224, 224),  # ImageNet standard
            (1, 3, 299, 299),  # Inception
            (1, 3, 256, 256),  # Common size
            (1, 3, 128, 128),  # Smaller size
            (1, 1, 224, 224),  # Grayscale
        ]
        
        successful_shape = None
        for shape in test_shapes:
            try:
                with torch.no_grad():
                    dummy_input = torch.randn(shape).to(device)
                    output = model(dummy_input)
                    print(f"✅ Input shape {shape} works!")
                    print(f"   Output shape: {output.shape}")
                    successful_shape = shape
                    break
            except Exception as e:
                print(f"❌ Input shape {shape} failed: {str(e)[:50]}")
        
        print()
        
        if successful_shape:
            print("Recommended Configuration:")
            print(f"  PYTORCH_INPUT_SIZE: {successful_shape[2:]}")
            print(f"  Input channels: {successful_shape[1]}")
            print(f"  Output classes: {output.shape[1] if len(output.shape) > 1 else 'Unknown'}")
            print()
            
            # Try to get class probabilities
            if len(output.shape) > 1:
                num_classes = output.shape[1]
                print(f"Number of output classes: {num_classes}")
                print()
                
                # Apply softmax to see probability distribution
                probs = torch.softmax(output, dim=1)
                print("Sample output probabilities (random input):")
                for i in range(min(num_classes, 15)):
                    print(f"  Class {i}: {probs[0][i].item():.4f}")
                if num_classes > 15:
                    print(f"  ... and {num_classes - 15} more classes")
        else:
            print("❌ Could not determine input shape")
            print("   Manual inspection of model architecture required")
        
        print()
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"❌ Error inspecting model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Default model path
    model_path = "models/optimizer.pt"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    inspect_model(model_path)
