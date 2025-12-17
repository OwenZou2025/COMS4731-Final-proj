#!/usr/bin/env python3
"""
Test script to verify the U2NET segmentation + ViT classification pipeline
Usage: python test_segmentation_pipeline.py <image_path>
"""

import sys
import os
import torch
from PIL import Image

# Add custom_ml_model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'custom_ml_model'))

from networks import U2NET
from cloth_segmentation import load_seg_model, generate_mask, get_palette, segmentation
from model_architecture import MultiTaskViT
from vit_llm_infer import load_model, predict

def test_pipeline(image_path):
    """Test the full segmentation + classification pipeline"""
    
    print("=" * 60)
    print("Testing U2NET Segmentation + ViT Classification Pipeline")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n1. Using device: {device}")
    
    # Create directories
    os.makedirs("./input/raw_images", exist_ok=True)
    os.makedirs("./input/masks", exist_ok=True)
    os.makedirs("./output/segmented_clothes", exist_ok=True)
    print("2. Created input and output directories")
    
    # Load U2NET segmentation model
    print("\n3. Loading U2NET segmentation model...")
    seg_checkpoint_path = "./custom_ml_model/model/checkpoint_u2net.pth"
    if not os.path.exists(seg_checkpoint_path):
        print(f"   ✗ Error: U2NET checkpoint not found at {seg_checkpoint_path}")
        return False
    
    seg_model = load_seg_model(seg_checkpoint_path, device=device)
    palette = get_palette(4)
    print("   ✓ U2NET model loaded successfully")
    
    # Load ViT classification model
    print("\n4. Loading ViT classification model...")
    vit_checkpoint_path = "./custom_ml_model/model/best_multitask_vit.pth"
    if not os.path.exists(vit_checkpoint_path):
        print(f"   ✗ Error: ViT checkpoint not found at {vit_checkpoint_path}")
        return False
    
    model_name = "google/vit-base-patch16-224-in21k"
    model, processor, label_encoders = load_model(vit_checkpoint_path, model_name, device)
    print("   ✓ ViT model loaded successfully")
    
    # Process test image
    print(f"\n5. Processing test image: {image_path}")
    if not os.path.exists(image_path):
        print(f"   ✗ Error: Image not found at {image_path}")
        return False
    
    # Save to raw_images
    import time
    img_name = f"test_{int(time.time() * 1000)}"
    raw_image_path = os.path.join("./input/raw_images", f"{img_name}.jpg")
    
    img = Image.open(image_path).convert("RGB")
    img.save(raw_image_path)
    print(f"   ✓ Saved to {raw_image_path}")
    
    # Run U2NET segmentation
    print("\n6. Running U2NET segmentation...")
    cloth_seg = generate_mask(img, net=seg_model, palette=palette, 
                             img_name=img_name, device=device)
    
    # Find generated mask and create segmented clothing
    mask_path = None
    for cls in [1, 2, 3]:
        potential_mask = os.path.join("./input/masks", f"{img_name}{cls}.png")
        if os.path.exists(potential_mask):
            mask_path = potential_mask
            print(f"   ✓ Mask generated: {mask_path}")
            break
    
    segmented_cloth_path = None
    if mask_path:
        print("\n7. Generating segmented clothing...")
        output_dir = "./output/segmented_clothes"
        segmentation(mask_path, raw_image_path, output_dir)
        
        # Check for generated segmented clothes (prefer white background)
        base_name = os.path.splitext(os.path.basename(mask_path))[0]
        potential_segmented = os.path.join(output_dir, f"{base_name}_white_bg.png")
        
        if os.path.exists(potential_segmented):
            segmented_cloth_path = potential_segmented
            print(f"   ✓ Segmented clothing: {segmented_cloth_path}")
        else:
            # Try transparent version as fallback
            potential_segmented = os.path.join(output_dir, f"{base_name}_transparent.png")
            if os.path.exists(potential_segmented):
                segmented_cloth_path = potential_segmented
                print(f"   ✓ Segmented clothing (transparent): {segmented_cloth_path}")
    
    if not segmented_cloth_path:
        print("   ⚠ Warning: No segmented clothing generated, will use original image")
        segmented_cloth_path = raw_image_path
    
    # Run ViT classification on segmented clothing
    print("\n8. Running ViT classification on segmented clothing...")
    predictions = predict(model, segmented_cloth_path, processor, label_encoders, device)
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    for task, label in predictions.items():
        print(f"  {task:15s}: {label}")
    
    print("\n" + "=" * 60)
    print("✓ Pipeline test completed successfully!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_segmentation_pipeline.py <image_path>")
        print("\nExample:")
        print("  python test_segmentation_pipeline.py path/to/shirt.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    success = test_pipeline(image_path)
    sys.exit(0 if success else 1)
