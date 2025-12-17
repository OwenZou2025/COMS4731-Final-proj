#!/usr/bin/env python3
"""
Flask server wrapper for the ML backend (U2NET segmentation + ViT model + Gemini LLM)
This allows the Node.js server to communicate with the Python ML pipeline.
"""

import os
import sys
import json
import tempfile
import numpy as np
import base64
from io import BytesIO
from typing import List, Dict
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from collections import OrderedDict

# Add custom_ml_model to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'custom_ml_model'))

from model_architecture import MultiTaskViT
from vit_llm_infer import (
    load_model, predict, detect_garment_type, build_vision_output
)
from transformers import ViTImageProcessor
from google import genai  # Using google-genai package
from networks import U2NET
from cloth_segmentation import load_seg_model, generate_mask, get_palette, segmentation, combine_masks
import cv2

app = Flask(__name__)
CORS(app)

# Global model variables (loaded once at startup)
model = None
processor = None
label_encoders = None
device = None
seg_model = None  # U2NET segmentation model
palette = None  # Color palette for segmentation

def init_model():
    """Initialize the ML models at server startup"""
    global model, processor, label_encoders, device, seg_model, palette
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load U2NET segmentation model
    seg_checkpoint_path = "./custom_ml_model/model/checkpoint_u2net.pth"
    print(f"Loading U2NET segmentation model from {seg_checkpoint_path}...")
    seg_model = load_seg_model(seg_checkpoint_path, device=device)
    palette = get_palette(4)
    print("U2NET segmentation model loaded successfully!")
    
    # Load ViT classification model
    checkpoint_path = "./custom_ml_model/model/best_multitask_vit.pth"
    model_name = "google/vit-base-patch16-224-in21k"
    print(f"Loading ViT model from {checkpoint_path}...")
    print(f"Using device: {device}")
    
    model, processor, label_encoders = load_model(checkpoint_path, model_name, device)
    print("ViT model loaded successfully!")
    
    # Create necessary directories
    os.makedirs("./input/raw_images", exist_ok=True)
    os.makedirs("./input/masks", exist_ok=True)
    os.makedirs("./output/segmented_clothes", exist_ok=True)
    print("Created input and output directories")

def image_to_base64(image_path: str) -> str:
    """
    Convert an image file to base64 string for frontend display
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save to bytes buffer
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Encode to base64
            img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def build_gemini_style_prompt(predictions_list: List[Dict], context: str) -> str:
    """Build Gemini-compatible prompt from ML predictions"""
    # Format items like Gemini does: "[idx] Name - Color: X, Texture: Y, Category: Z"
    attributes_text = ""
    for idx, pred in enumerate(predictions_list):
        name = f"{pred.get('gender', '')} {pred.get('baseColour', '')} {pred.get('articleType', '')}".strip().title()
        color = pred.get('baseColour', 'Unknown').title()
        texture = "Unknown"  # ViT doesn't predict texture
        category = pred.get('usage', 'Casual').title()
        
        attributes_text += f"[{idx}] {name} - Color: {color}, Texture: {texture}, Category: {category}\n"
    
    prompt = f"""You are a professional fashion stylist. I have the following clothing items:
{attributes_text}

The occasion/context is: {context}

Please analyze and provide a response in TWO parts:

PART 1 - Selected Items (JSON format):
Return a JSON array of item indices (the numbers in brackets like [0], [1], etc.) that are appropriate for this occasion.
Example: [0, 2, 3]

PART 2 - Style Recommendation (text):
Provide a detailed style assessment organized into these EXACT subsections.

CRITICAL NAMING RULE: NEVER use "Item 0", "Item 1", "Item 2" etc. NEVER write formats like "Item 0 (description):" or "Item 1 (name):". Instead, ALWAYS refer to items ONLY by their descriptive names. ALWAYS use Title Case for item names (capitalize each word), like "Navy Straight-leg Trousers", "Double-breasted Plaid Blazer", or "Cropped Tweed Jacket".

**Overall Assessment**
A brief style assessment of the SELECTED items (2-3 sentences about the overall look and aesthetic)

**Why These Pieces Work**
For each selected item, explain why it works for this occasion. Format as bullet points with the item name in BOLD (in Title Case) followed by a colon, like:
- **Navy Straight-leg Trousers:** These are a foundational piece...
- **Double-breasted Plaid Blazer:** This adds sophistication...

**Outfit Combinations**
Provide 2-3 specific outfit combination suggestions. Give each outfit a CREATIVE NAME in bold, then describe what items to combine. Format like:
1. **The Polished Professional:** Pair the Navy Straight-leg Trousers with...
2. **Chic Business Casual:** Combine the Double-breasted Plaid Blazer with...
3. **Effortless Elegance:** Layer the Cropped Tweed Jacket over...

**Additional Styling Tips**
Organize tips by CATEGORY with bold headers. Format like:
- **Tops:** Suggestions for tops to pair with these items...
- **Footwear:** Shoe recommendations...
- **Accessories:** Belt, jewelry, bag suggestions...
- **Layering:** Tips for layering pieces...

Format your response exactly as:
SELECTED_ITEMS: [array of indices]
RECOMMENDATION: your text here

Keep the recommendation natural, friendly, and conversational. Make sure to include all four subsection headers in bold (wrapped in **)."""
    
    return prompt

def parse_gemini_response(text: str, total_items: int) -> tuple:
    """Parse Gemini-style response to extract selected items and recommendation"""
    import re
    
    selected_items = []
    recommendation = text
    
    # Try to extract selected items
    selected_match = re.search(r'SELECTED_ITEMS:\s*(\[[^\]]+\])', text)
    if selected_match:
        try:
            selected_items = json.loads(selected_match.group(1))
            # Validate indices
            selected_items = [idx for idx in selected_items if 0 <= idx < total_items]
        except:
            print('Could not parse selected items, selecting all')
    
    # Extract recommendation text
    rec_match = re.search(r'RECOMMENDATION:\s*([\s\S]*)', text)
    if rec_match:
        recommendation = rec_match.group(1).strip()
    
    # If no items selected, select all
    if not selected_items:
        selected_items = list(range(total_items))
    
    return recommendation, selected_items

def format_attribute_for_frontend(predictions: Dict[str, str], image_index: int, mask_path: str = None) -> Dict:
    """
    Convert ML backend predictions to Gemini-like format for frontend compatibility
    """
    article_type = predictions.get('articleType', 'Unknown')
    base_colour = predictions.get('baseColour', 'Unknown')
    gender = predictions.get('gender', 'Unisex')
    usage = predictions.get('usage', 'Casual')
    
    # Create a name similar to Gemini's format
    name = f"{gender} {base_colour} {article_type}"
    
    result = {
        "isClothing": True,
        "name": name.title(),
        "color": base_colour.title(),
        "texture": "Unknown",  # ViT doesn't predict texture
        "category": article_type.title(),
        "confidence": 0.85,  # Mock confidence since ViT doesn't provide it
        # Include original ML predictions for LLM
        "mlPredictions": predictions
    }
    
    # Add segmented clothing image data if available
    if mask_path and os.path.exists(mask_path):
        segmented_base64 = image_to_base64(mask_path)
        if segmented_base64:
            result["maskImage"] = segmented_base64  # Keep field name for frontend compatibility
    
    return result

@app.route('/api/ml/analyze-images', methods=['POST'])
def analyze_images():
    """
    Analyze images using U2NET segmentation + ViT model
    Pipeline: Upload → Save to raw_images → U2NET segmentation → Generate segmented clothes → ViT classification
    Expected: multipart/form-data with 'images' field containing image files
    """
    try:
        if 'images' not in request.files:
            return jsonify({"error": "No images provided"}), 400
        
        files = request.files.getlist('images')
        if not files:
            return jsonify({"error": "No images provided"}), 400
        
        print(f"Analyzing {len(files)} images with ML backend (U2NET + ViT)...")
        
        attributes = []
        raw_image_paths = []
        mask_paths = []
        
        try:
            # Step 1: Save uploaded files to input/raw_images and run U2NET segmentation
            for idx, file in enumerate(files):
                # Generate unique filename
                import time
                img_name = f"img_{int(time.time() * 1000)}_{idx}"
                raw_image_path = os.path.join("./input/raw_images", f"{img_name}.jpg")
                
                # Save original image
                file.save(raw_image_path)
                raw_image_paths.append(raw_image_path)
                
                print(f"Processing image {idx + 1}/{len(files)}: {file.filename}")
                
                # Step 2: Run U2NET segmentation
                print(f"  Running U2NET segmentation...")
                img = Image.open(raw_image_path).convert("RGB")
                cloth_seg = generate_mask(img, net=seg_model, palette=palette, 
                                        img_name=img_name, device=device)
                
                # Step 3: Process ALL mask classes separately (upper, middle, lower regions)
                # Collect all available masks (classes 1, 2, 3)
                all_mask_paths = []
                for cls in [1, 2, 3]:
                    potential_mask = os.path.join("./input/masks", f"{img_name}{cls}.png")
                    if os.path.exists(potential_mask):
                        all_mask_paths.append((cls, potential_mask))
                
                if all_mask_paths:
                    print(f"  Found {len(all_mask_paths)} clothing regions, processing separately...")
                    
                    output_dir = "./output/segmented_clothes"
                    
                    # Process each mask separately and classify each region
                    for cls, mask_path in all_mask_paths:
                        # Use white background only (faster, smaller files)
                        segmentation(mask_path, raw_image_path, output_dir, white_bg_only=True)
                        
                        # Check for generated segmented clothing
                        base_name = os.path.splitext(os.path.basename(mask_path))[0]
                        segmented_path = os.path.join(output_dir, f"{base_name}_white_bg.png")
                        
                        if os.path.exists(segmented_path):
                            print(f"    Region {cls}: {segmented_path}")
                            
                            # Step 4: Run ViT inference on THIS segmented region
                            print(f"    Classifying region {cls}...")
                            predictions = predict(model, segmented_path, processor, label_encoders, device)
                            
                            # Format for frontend (include segmented clothing image)
                            attr = format_attribute_for_frontend(predictions, len(attributes), segmented_path)
                            attr['region'] = cls  # Add region identifier
                            attributes.append(attr)
                            
                            print(f"    Predictions: {predictions['articleType']}, {predictions['baseColour']}")
                else:
                    print(f"  Warning: No masks generated, using original image")
                    # Fallback to original image
                    predictions = predict(model, raw_image_path, processor, label_encoders, device)
                    attr = format_attribute_for_frontend(predictions, idx, None)
                    attributes.append(attr)
        
        finally:
            # Clean up: Remove temporary files from input directories
            # Note: In production, you may want to keep these for debugging
            # or implement a cleanup job that runs periodically
            pass
        
        print(f"\n=== SUMMARY ===")
        print(f"Total segments processed: {len(attributes)}")
        for i, attr in enumerate(attributes):
            print(f"  [{i}] Region {attr.get('region', '?')}: {attr.get('category', 'Unknown')}")
        print(f"===============\n")
        
        return jsonify({
            "success": True,
            "attributes": attributes,
            "count": len(attributes)
        })
        
    except Exception as e:
        print(f"Error in analyze-images: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Failed to analyze images",
            "message": str(e)
        }), 500

@app.route('/api/ml/analyze-wardrobe', methods=['POST'])
def analyze_wardrobe():
    """
    Analyze wardrobe images using U2NET segmentation + ViT model and get outfit recommendation
    Pipeline: Base64 decode → Save to raw_images → U2NET segmentation → Generate segmented clothes → ViT classification → LLM recommendation
    Expected JSON: { "wardrobeItems": [{id, imageData}], "context": "..." }
    This uses U2NET for segmentation, vit_infer.py for analysis and vit_llm_infer.py for recommendations
    """
    try:
        data = request.json
        if not data or 'wardrobeItems' not in data or 'context' not in data:
            return jsonify({"error": "Missing wardrobeItems or context"}), 400
        
        wardrobe_items = data['wardrobeItems']
        context = data['context']
        
        if not wardrobe_items:
            return jsonify({"error": "No wardrobe items provided"}), 400
        
        print(f"Analyzing {len(wardrobe_items)} wardrobe items for context: {context}")
        
        # Step 1: Run U2NET segmentation + ViT inference on each wardrobe image
        import base64
        import time
        
        raw_image_paths = []
        mask_paths = []
        predictions_list = []
        
        try:
            for idx, item in enumerate(wardrobe_items):
                # Decode base64 image data
                image_data = item['imageData']
                if ',' in image_data:
                    # Remove data:image/...;base64, prefix
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                
                # Generate unique filename
                img_name = f"wardrobe_{int(time.time() * 1000)}_{idx}"
                raw_image_path = os.path.join("./input/raw_images", f"{img_name}.jpg")
                
                # Save original image
                with open(raw_image_path, 'wb') as f:
                    f.write(image_bytes)
                raw_image_paths.append(raw_image_path)
                
                print(f"  Analyzing wardrobe item {idx + 1}/{len(wardrobe_items)}...")
                
                # Step 2: Run U2NET segmentation
                print(f"    Running U2NET segmentation...")
                img = Image.open(raw_image_path).convert("RGB")
                cloth_seg = generate_mask(img, net=seg_model, palette=palette, 
                                        img_name=img_name, device=device)
                
                # Step 3: Process ALL mask classes separately (upper, middle, lower regions)
                all_mask_paths = []
                for cls in [1, 2, 3]:
                    potential_mask = os.path.join("./input/masks", f"{img_name}{cls}.png")
                    if os.path.exists(potential_mask):
                        all_mask_paths.append((cls, potential_mask))
                
                if all_mask_paths:
                    print(f"    Found {len(all_mask_paths)} clothing regions, processing separately...")
                    
                    output_dir = "./output/segmented_clothes"
                    
                    # Process each mask separately and classify each region
                    for cls, mask_path in all_mask_paths:
                        # Use white background only (faster, smaller files)
                        segmentation(mask_path, raw_image_path, output_dir, white_bg_only=True)
                        
                        # Check for generated segmented clothing
                        base_name = os.path.splitext(os.path.basename(mask_path))[0]
                        segmented_path = os.path.join(output_dir, f"{base_name}_white_bg.png")
                        
                        if os.path.exists(segmented_path):
                            print(f"      Region {cls}: {segmented_path}")
                            
                            # Run ViT inference on THIS segmented region
                            print(f"      Classifying region {cls}...")
                            preds = predict(model, segmented_path, processor, label_encoders, device)
                            preds['wardrobe_item_id'] = item.get('id')
                            preds['index'] = len(predictions_list)
                            preds['region'] = cls
                            predictions_list.append(preds)
                            
                            print(f"      Predictions: {preds['articleType']}, {preds['baseColour']}")
                else:
                    print(f"    Warning: No masks generated, using original image")
                    # Fallback to original image
                    preds = predict(model, raw_image_path, processor, label_encoders, device)
                    preds['wardrobe_item_id'] = item.get('id')
                    preds['index'] = idx
                    predictions_list.append(preds)
        
        finally:
            # Clean up: keep files for debugging or implement periodic cleanup
            pass
        
        # Step 2: Use vit_llm_infer.py logic to get outfit recommendation
        
        # Build vision output (categorize by tops/bottoms/onepieces)
        vision_output = build_vision_output(predictions_list, context)
        print(f"\nVision output: {len(vision_output['tops'])} tops, "
              f"{len(vision_output['bottoms'])} bottoms, "
              f"{len(vision_output['onepieces'])} onepieces")
        
        # Get LLM recommendation if API key available
        if 'GOOGLE_API_KEY' in os.environ or 'GEMINI_API_KEY' in os.environ:
            api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
            client = genai.Client(api_key=api_key)
            
            # Try models in order of preference (different rate limit pools)
            # Can be overridden with GEMINI_MODEL env variable
            preferred_model = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-exp')
            
            # Build fallback list starting with preferred model
            models_to_try = [preferred_model]
            # Add other models as fallbacks if not already in list
            # Note: These are verified model names from the API
            for model_name in ["gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-flash-latest"]:
                if model_name not in models_to_try:
                    models_to_try.append(model_name)
            
            recommendation_text = None
            selected_indices = None
            
            for model_name in models_to_try:
                try:
                    # Build Gemini-style prompt
                    prompt = build_gemini_style_prompt(predictions_list, context)
                    
                    print(f"  Trying {model_name}...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    
                    # Parse response in Gemini format
                    recommendation_text, selected_indices = parse_gemini_response(
                        response.text, len(predictions_list)
                    )
                    
                    print(f"  ✓ Success with {model_name}")
                    break  # Success - exit the loop
                    
                except Exception as api_error:
                    error_str = str(api_error)
                    if 'RESOURCE_EXHAUSTED' in error_str or '429' in error_str or 'quota' in error_str.lower():
                        print(f"  ✗ {model_name} rate limited, trying next model...")
                        continue  # Try next model
                    else:
                        # Non-rate-limit error - re-raise
                        raise
            
            # If we got a response from any model, return it
            if recommendation_text is not None:
                return jsonify({
                    "success": True,
                    "recommendation": recommendation_text,
                    "selectedItems": selected_indices,
                    "visionOutput": vision_output,
                    "predictions": predictions_list
                })
            else:
                # All models hit rate limits
                print("  All models rate limited, falling back to basic recommendation")
        
        # No API key or rate limit hit - provide basic recommendation
        recommendation_text = format_wardrobe_basic_recommendation(predictions_list, context)
        selected_indices = list(range(min(2, len(predictions_list))))
        
        return jsonify({
            "success": True,
            "recommendation": recommendation_text,
            "selectedItems": selected_indices,
            "visionOutput": vision_output,
            "predictions": predictions_list
        })
    
    except Exception as e:
        print(f"Error in analyze-wardrobe: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Failed to analyze wardrobe",
            "message": str(e)
        }), 500

@app.route('/api/ml/get-recommendation', methods=['POST'])
def get_recommendation():
    """
    Get outfit recommendation using ML predictions + Gemini LLM
    Expected JSON: { "attributes": [...], "context": "..." }
    """
    try:
        data = request.json
        if not data or 'attributes' not in data or 'context' not in data:
            return jsonify({"error": "Missing attributes or context"}), 400
        
        attributes = data['attributes']
        context = data['context']
        
        print(f"Getting recommendation for context: {context}")
        print(f"Number of items: {len(attributes)}")
        
        # Extract ML predictions from attributes
        items = []
        for attr in attributes:
            if 'mlPredictions' in attr:
                items.append(attr['mlPredictions'])
            else:
                # Fallback: construct from frontend attributes
                items.append({
                    'gender': attr.get('name', '').split()[0] if attr.get('name') else 'Unisex',
                    'articleType': attr.get('category', 'Unknown'),
                    'baseColour': attr.get('color', 'Unknown'),
                    'season': 'All',
                    'usage': 'Casual'
                })
        
        # Build vision output for LLM
        vision_output = build_vision_output(items, context)
        print(f"Vision output: tops={len(vision_output['tops'])}, "
              f"bottoms={len(vision_output['bottoms'])}, "
              f"onepieces={len(vision_output['onepieces'])}")
        
        # Get LLM recommendation if API key is available
        if 'GOOGLE_API_KEY' in os.environ or 'GEMINI_API_KEY' in os.environ:
            api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
            client = genai.Client(api_key=api_key)
            
            # Try models in order of preference (different rate limit pools)
            # Can be overridden with GEMINI_MODEL env variable
            preferred_model = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-exp')
            
            # Build fallback list starting with preferred model
            models_to_try = [preferred_model]
            # Add other models as fallbacks if not already in list
            # Note: These are verified model names from the API
            for model_name in ["gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-flash-latest"]:
                if model_name not in models_to_try:
                    models_to_try.append(model_name)
            
            recommendation_text = None
            selected_items = None
            
            for model_name in models_to_try:
                try:
                    # Build Gemini-style prompt
                    prompt = build_gemini_style_prompt(items, context)
                    
                    print(f"  Trying {model_name}...")
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    
                    # Parse response in Gemini format
                    recommendation_text, selected_items = parse_gemini_response(
                        response.text, len(items)
                    )
                    
                    print(f"  ✓ Success with {model_name}")
                    break  # Success - exit the loop
                    
                except Exception as api_error:
                    error_str = str(api_error)
                    if 'RESOURCE_EXHAUSTED' in error_str or '429' in error_str or 'quota' in error_str.lower():
                        print(f"  ✗ {model_name} rate limited, trying next model...")
                        continue  # Try next model
                    else:
                        # Non-rate-limit error - re-raise
                        raise
            
            # If we got a response from any model, return it
            if recommendation_text is not None:
                return jsonify({
                    "success": True,
                    "recommendation": recommendation_text,
                    "selectedItems": selected_items
                })
            else:
                # All models hit rate limits
                print("  All models rate limited, falling back to basic recommendation")
        
        # No API key or rate limit hit - provide basic recommendation
        recommendation_text = f"Based on ML analysis for {context}:\n\n"
        recommendation_text += format_basic_recommendation(items, context)
        
        return jsonify({
            "success": True,
            "recommendation": recommendation_text,
            "selectedItems": list(range(min(2, len(attributes))))
        })
        
    except Exception as e:
        print(f"Error in get-recommendation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Failed to get recommendation",
            "message": str(e)
        }), 500


def format_basic_recommendation(items: List[Dict], context: str) -> str:
    """Provide a basic recommendation without LLM"""
    text = "Items detected:\n"
    for idx, item in enumerate(items, 1):
        text += f"{idx}. {item.get('gender', '')} {item.get('baseColour', '')} "
        text += f"{item.get('articleType', '')} - {item.get('usage', '')}\n"
    
    text += f"\nSuggestion: Consider the formality required for '{context}' "
    text += "when selecting your outfit. Match colors and styles appropriately."
    
    return text


def format_wardrobe_basic_recommendation(predictions_list: List[Dict], context: str) -> str:
    """Provide a basic wardrobe recommendation without LLM"""
    from vit_llm_infer import detect_garment_type
    
    text = f"**Wardrobe Analysis for {context}:**\n\n"
    text += "Items in your selection:\n"
    
    for idx, pred in enumerate(predictions_list, 1):
        garment_type = detect_garment_type(pred.get('articleType', ''))
        text += f"{idx}. {pred.get('gender', '')} {pred.get('baseColour', '')} "
        text += f"{pred.get('articleType', '')} ({garment_type or 'other'}) - {pred.get('usage', '')}\n"
    
    text += f"\n**Suggestion:**\nConsider pairing complementary items based on the formality "
    text += f"required for '{context}'. Match colors and styles for a cohesive look."
    
    return text


@app.route('/api/ml/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "message": "ML Backend is running",
        "device": str(device),
        "model_loaded": model is not None
    })

if __name__ == '__main__':
    print("Initializing ML Backend Server...")
    init_model()
    
    port = int(os.environ.get('ML_PORT', 5001))  # Default to 5001 to avoid macOS AirPlay conflict
    print(f"\nML Backend Server starting on port {port}...")
    print(f"Health check: http://localhost:{port}/api/ml/health\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
