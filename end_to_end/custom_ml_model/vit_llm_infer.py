import sys
import os
import json
from typing import Literal, Optional, List, Dict

import torch
from torch.serialization import add_safe_globals
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel
from google import genai
from transformers import ViTImageProcessor

from model_architecture import MultiTaskViT


add_safe_globals([LabelEncoder])


TOPS = {
    "Shirts", "Tshirts", "Tops", "Sweatshirts", "Sweaters", "Jackets",
    "Blazers", "Shrug", "Tunics", "Kurtas", "Waistcoat", "Innerwear Vests",
    "Camisoles", "Lounge Tshirts", "Nehru Jackets", "Robe", "Rain Jacket",
    "Bath Robe", "Nightdress", "Baby Dolls"
}

BOTTOMS = {
    "Jeans", "Trousers", "Track Pants", "Shorts", "Skirts", "Capris",
    "Leggings", "Jeggings", "Boxers", "Briefs", "Trunk", "Lounge Pants",
    "Tights", "Stockings", "Rain Trousers", "Salwar", "Patiala", "Churidar"
}

ONEPIECES = {
    "Dresses", "Jumpsuit", "Rompers", "Kurta Sets", "Clothing Set",
    "Lehenga Choli", "Night suits", "Suits", "Salwar and Dupatta"
}


class OutfitSelection(BaseModel):
    outfit_type: Literal["top_bottom", "onepiece"]
    selected_top: Optional[str] = None
    selected_bottom: Optional[str] = None
    selected_onepiece: Optional[str] = None
    style: Literal[
        "formal", "business_casual", "smart_casual", "casual",
        "loungewear", "sporty", "streetwear", "evening_elegant",
        "cultural", "other"
    ]
    fit: str
    reason: str


def load_model(checkpoint_path: str, model_name: str, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    label_encoders = checkpoint["label_encoders"]
    num_classes = checkpoint["num_classes"]

    model = MultiTaskViT(model_name, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    processor = ViTImageProcessor.from_pretrained(model_name)

    return model, processor, label_encoders


def predict(model, image_path: str, processor, label_encoders: Dict[str, LabelEncoder], device: torch.device) -> Dict[str, str]:
    model.eval()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values)

    predictions = {}
    for task in ["gender", "articleType", "baseColour", "season", "usage"]:
        _, predicted_idx = torch.max(outputs[task], 1)
        predicted_label = label_encoders[task].inverse_transform([predicted_idx.item()])[0]
        predictions[task] = predicted_label

    return predictions


def detect_garment_type(article_type: str) -> Optional[str]:
    if article_type in TOPS:
        return "top"
    if article_type in BOTTOMS:
        return "bottom"
    if article_type in ONEPIECES:
        return "onepiece"
    return None


def build_vision_output(items: List[Dict[str, str]], event: str) -> Dict[str, object]:
    vision_output = {"tops": [], "bottoms": [], "onepieces": [], "event": event}
    for item in items:
        t = detect_garment_type(item.get("articleType", ""))
        if t == "top":
            vision_output["tops"].append(item)
        elif t == "bottom":
            vision_output["bottoms"].append(item)
        elif t == "onepiece":
            vision_output["onepieces"].append(item)
    return vision_output


def build_prompt(vision_output: dict) -> str:
    input_json = json.dumps(vision_output, ensure_ascii=False, indent=2)
    prompt = f"""
You are a professional fashion stylist AI assistant.

Each garment is described only by:
gender, articleType, baseColour, season, usage.

Input structure:
- "tops": a list of separate upper garments (may be empty)
- "bottoms": a list of separate lower garments (may be empty)
- "onepieces": a list of one-piece outfits such as dresses or jumpsuits (may be empty)
- "event": the occasion.

You must choose exactly ONE outfit in one of these forms:

1) A top–bottom combination:
   - one top from "tops"
   - one bottom from "bottoms"
   In this case:
   - "outfit_type" = "top_bottom"
   - "selected_top" = a short natural phrase for the chosen top
   - "selected_bottom" = a short natural phrase for the chosen bottom
   - "selected_onepiece" = null

2) A single one-piece outfit:
   - one item from "onepieces"
   In this case:
   - "outfit_type" = "onepiece"
   - "selected_onepiece" = a short natural phrase for the chosen one-piece item
   - "selected_top" = null
   - "selected_bottom" = null

If "onepieces" is non-empty, you are allowed to choose either:
- the best one-piece item, or
- the best top–bottom combination,
whichever fits the event better.

If "tops" and "bottoms" are both empty, but "onepieces" is not empty, you MUST choose a "onepiece".
If "onepieces" is empty but at least one top and one bottom exist, you MUST choose a "top_bottom" outfit.

CRITICAL PAIRING RULE: Tops and bottoms ALWAYS come in pairs - you cannot select one without the other:
- If you select a top, you MUST also select a bottom
- If you select a bottom, you MUST also select a top
- NEVER select a top alone or a bottom alone
- If you cannot find a suitable pairing, choose a one-piece outfit instead (if available)
This is a strict requirement - incomplete outfits are not acceptable.

Classify the outfit's style as one of:
["formal","business_casual","smart_casual","casual",
 "loungewear","sporty","streetwear","evening_elegant",
 "cultural","other"].

Use the following guidelines when interpreting the item-level “usage” labels.
These are NOT strict rules — they only indicate tendencies. Always consider the event,
the combination of garments, fabric type, silhouette, color, and level of refinement.

- usage = "Formal" → likely "formal", "evening_elegant", or "business_casual".
- usage = "Smart Casual" → likely "smart_casual" or "business_casual".
- usage = "Casual" → likely "casual", sometimes "smart_casual" if refined.
- usage = "Sports" → likely "sporty".
- usage = "Party" → may be "evening_elegant", "streetwear", or "other" depending on design.
- usage = "Travel" or "Home" → likely "casual" or "loungewear".
- usage = "Ethnic" → likely "cultural", or "evening_elegant" when fabrics/embellishments are refined.

Please analyze and provide a response in TWO parts:

PART 1 - Selected Items (JSON format):
Return a JSON object with these keys:
- "outfit_type": either "top_bottom" or "onepiece"
- "selected_indices": object with structure:
  - If outfit_type is "top_bottom": {{"tops": [index], "bottoms": [index]}} - specify which index is the top and which is the bottom
  - If outfit_type is "onepiece": {{"onepieces": [index]}} - specify the onepiece index
  Example for top_bottom: {{"tops": [0], "bottoms": [2]}}
  Example for onepiece: {{"onepieces": [1]}}
- "style": one of ["formal","business_casual","smart_casual","casual","loungewear","sporty","streetwear","evening_elegant","cultural","other"]
- "fit": one of ["appropriate","too casual","too formal","inappropriate"]

PART 2 - Style Recommendation (text):
Provide a detailed style assessment organized into these EXACT subsections.

CRITICAL NAMING RULE: NEVER use "Item 0", "Item 1", "Item 2" etc. NEVER write formats like "Item 0 (description):" or "Item 1 (name):". Instead, ALWAYS refer to items ONLY by their descriptive names formed from the attributes (gender, baseColour, articleType). ALWAYS use Title Case for item names (capitalize each word), like "Navy Straight-leg Trousers", "White Formal Shirt", or "Black Party Dress".

**Overall Assessment**
A brief style assessment of the SELECTED items (2-3 sentences about the overall look and aesthetic)

**Why These Pieces Work**
For each selected item, explain why it works for this occasion. Format as bullet points with the item name in BOLD (in Title Case) followed by a colon, like:
- **Navy Straight-leg Trousers:** These are a foundational piece...
- **White Formal Shirt:** This adds sophistication...

**Outfit Combinations**
Provide 2-3 specific outfit combination suggestions. Give each outfit a CREATIVE NAME in bold, then describe what items to combine. Format like:
1. **The Polished Professional:** Pair the Navy Straight-leg Trousers with...
2. **Chic Business Casual:** Combine the White Formal Shirt with...
3. **Effortless Elegance:** Layer the items together for...

**Additional Styling Tips**
Organize tips by CATEGORY with bold headers. Format like:
- **Footwear:** Shoe recommendations...
- **Accessories:** Belt, jewelry, bag suggestions...
- **Layering:** Tips for layering pieces...

Format your response exactly as:
SELECTED_ITEMS: {{{{json object}}}}
RECOMMENDATION: your detailed text here

Keep the recommendation natural, friendly, and conversational. Make sure to include all four subsection headers in bold (wrapped in **).

Input:
{input_json}
"""
    return prompt


def parse_llm_response(text: str) -> tuple:
    """Parse LLM response to extract selected items JSON and recommendation text"""
    import re
    
    selected_data = None
    recommendation = text
    
    # Try to extract selected items JSON (handles nested objects)
    selected_match = re.search(r'SELECTED_ITEMS:\s*(\{[^}]*\{[^}]*\}[^}]*\}|\{[^}]+\})', text)
    if selected_match:
        try:
            selected_data = json.loads(selected_match.group(1))
        except Exception as e:
            print(f'Warning: Could not parse selected items JSON: {e}')
            # Try a more aggressive search for nested JSON
            try:
                # Find content between SELECTED_ITEMS: and RECOMMENDATION:
                content_match = re.search(r'SELECTED_ITEMS:\s*(.*?)\s*RECOMMENDATION:', text, re.DOTALL)
                if content_match:
                    json_str = content_match.group(1).strip()
                    selected_data = json.loads(json_str)
            except Exception as e2:
                print(f'Warning: Fallback parsing also failed: {e2}')
    
    # Validate top-bottom pairing
    if selected_data and 'selected_indices' in selected_data:
        indices = selected_data['selected_indices']
        if isinstance(indices, dict):
            has_top = 'tops' in indices and indices['tops']
            has_bottom = 'bottoms' in indices and indices['bottoms']
            
            # Enforce pairing rule
            if has_top and not has_bottom:
                print('Warning: Top selected without bottom - this violates pairing rule')
                # Remove the incomplete selection
                selected_data['selected_indices'] = {}
            elif has_bottom and not has_top:
                print('Warning: Bottom selected without top - this violates pairing rule')
                # Remove the incomplete selection
                selected_data['selected_indices'] = {}
    
    # Extract recommendation text
    rec_match = re.search(r'RECOMMENDATION:\s*([\s\S]*)', text)
    if rec_match:
        recommendation = rec_match.group(1).strip()
    
    return selected_data, recommendation


def main():
    if len(sys.argv) < 3:
        print("Usage: python vit_llm_infer.py <user input> <image_path1> [<image_path2> ...]")
        sys.exit(1)

    event = sys.argv[1]
    image_paths = sys.argv[2:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "./model/best_multitask_vit.pth"
    model_name = "google/vit-base-patch16-224-in21k"

    model, processor, label_encoders = load_model(checkpoint_path, model_name, device)

    items = []
    for image_path in image_paths:
        print(f"Running vision inference on: {image_path}")
        preds = predict(model, image_path, processor, label_encoders, device)
        preds_with_id = dict(preds)
        preds_with_id["image_path"] = image_path
        items.append(preds_with_id)
        print("Predictions for this image:")
        for k, v in preds.items():
            print(f"  {k}: {v}")
        print()

    vision_output = build_vision_output(items, event)
    print("Constructed vision_output:")
    print(json.dumps(vision_output, ensure_ascii=False, indent=2))

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"]) # Set your API key in the environment variable
    prompt = build_prompt(vision_output)

    # Use text-based generation instead of structured JSON
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    # Parse the text response
    selected_data, recommendation = parse_llm_response(response.text)

    print("\n" + "="*60)
    print("LLM STYLING RECOMMENDATION")
    print("="*60)
    
    if selected_data:
        print("\n[Selection Details]")
        print(f"Outfit type: {selected_data.get('outfit_type', 'N/A')}")
        print(f"Style: {selected_data.get('style', 'N/A')}")
        print(f"Fit: {selected_data.get('fit', 'N/A')}")
        print(f"Selected indices: {selected_data.get('selected_indices', [])}")
    
    print("\n[Detailed Recommendation]")
    print(recommendation)
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
