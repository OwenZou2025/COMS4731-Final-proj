import sys
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image
from model_architecture import MultiTaskViT
from transformers import ViTModel, ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset, DatasetDict, Image, Features, ClassLabel, Value

def load_model(checkpoint_path, model_name, device):

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_encoders = checkpoint['label_encoders']
    num_classes = checkpoint['num_classes']

    model = MultiTaskViT(model_name, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    processor = ViTImageProcessor.from_pretrained(model_name)

    return model, processor, label_encoders, num_classes

def predict(model, image_path, processor, label_encoders, device):
    model.eval()

    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    with torch.no_grad():
        outputs = model(pixel_values)

    predictions = {}
    for task in ['gender', 'articleType', 'baseColour', 'season', 'usage']:
        _, predicted_idx = torch.max(outputs[task], 1)
        predicted_label = label_encoders[task].inverse_transform([predicted_idx.item()])[0]
        predictions[task] = predicted_label

    return predictions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]   # <-- user-provided image path
    print(f"Running inference on: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "./model/best_multitask_vit.pth"
    model_name = "google/vit-base-patch16-224-in21k"

    model, processor, label_encoders = load_model(checkpoint_path, model_name, device)

    preds = predict(model, image_path, processor, label_encoders, device)

    print("\nPredictions:")
    for task, label in preds.items():
        print(f"{task}: {label}")
