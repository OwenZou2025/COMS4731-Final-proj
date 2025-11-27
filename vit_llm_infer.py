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

from train import MultiTaskViT


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

Classify the outfit’s style as one of:
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

Evaluate the fit as one of:
["appropriate","too casual","too formal","inappropriate"].

Explain briefly (max 100 words).

You must output only valid JSON with the keys:
outfit_type, selected_top, selected_bottom, selected_onepiece, style, fit, reason.

If none of the possible outfits are appropriate, choose the least mismatched one and set fit accordingly.

Describe garments using short natural phrases such as:
"Men white formal shirt", "Women black casual jeans", "Women black party dress".

Input:
{input_json}
"""
    return prompt


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

    response = client.models.generate_content(
        model="models/gemini-2.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": OutfitSelection,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
        },
    )

    analysis: OutfitSelection = response.parsed

    print("\nLLM Styling Decision:")
    print("Outfit type:", analysis.outfit_type)
    if analysis.outfit_type == "onepiece":
        print("Selected Onepiece:", analysis.selected_onepiece)
    else:
        print("Selected Top:", analysis.selected_top)
        print("Selected Bottom:", analysis.selected_bottom)
    print("Style:", analysis.style)
    print("Fit:", analysis.fit)
    print("Reason:", analysis.reason)


if __name__ == "__main__":
    main()
