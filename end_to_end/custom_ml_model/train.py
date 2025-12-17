from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, Image, Features, ClassLabel, Value
import zipfile

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, datasets
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Import model architecture
from model_architecture import MultiTaskViT


#Please download datasets through the command: kaggle datasets download -d paramaggarwal/fashion-product-images-small
zip_ref = zipfile.ZipFile('fashion-product-images-small.zip', 'r')
zip_ref.extractall('/datasets')
zip_ref.close()
#########################################################################################

def prepare_data(df):
    label_encoders = {}
    num_classes = {}

    for col in ['gender', 'articleType', 'baseColour', 'season', 'usage']:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        label_encoders[col] = le
        num_classes[col] = len(le.classes_)
        print(f"{col}: {num_classes[col]} classes")

    return label_encoders, num_classes

def train_epoch(model, dataloader, optimizer, device, task_weights=None):
    model.train()
    total_loss = 0

    if task_weights is None:
        task_weights = {
            'gender': 1.0,
            'articleType': 1.0,
            'baseColour': 1.0,
            'season': 1.0,
            'usage': 1.0
        }

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)

        outputs = model(images)
        loss = 0
        for task in ['gender', 'articleType', 'baseColour', 'season', 'usage']:
            task_labels = labels[task].to(device)
            task_loss = criterion(outputs[task], task_labels)
            loss += task_weights[task] * task_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = {task: 0 for task in ['gender', 'articleType', 'baseColour', 'season', 'usage']}
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)

            batch_size = images.size(0)
            total += batch_size

            for task in ['gender', 'articleType', 'baseColour', 'season', 'usage']:
                task_labels = labels[task].to(device)
                _, predicted = torch.max(outputs[task], 1)
                correct[task] += (predicted == task_labels).sum().item()

    accuracies = {task: correct[task] / total for task in correct.keys()}
    return accuracies

class FashionDataset(Dataset):
    def __init__(self, df, image_dir, processor, label_encoders):
        self.df = df
        self.image_dir = image_dir
        self.processor = processor
        self.label_encoders = label_encoders

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = f"{self.image_dir}/{row['id']}.jpg"  # Adjust path as needed
        image = Image.open(img_path).convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)

        labels = {
            'gender': self.label_encoders['gender'].transform([row['gender']])[0],
            'articleType': self.label_encoders['articleType'].transform([row['articleType']])[0],
            'baseColour': self.label_encoders['baseColour'].transform([row['baseColour']])[0],
            'season': self.label_encoders['season'].transform([row['season']])[0],
            'usage': self.label_encoders['usage'].transform([row['usage']])[0],
        }

        return pixel_values, labels


def check_image_exists(image_filename):
    """
    Checks if the desired filename exists within the filenames found in the given directory.
    Returns True if the filename exists, False otherwise.
    """
    global images
    if image_filename in images:
        return image_filename
    else:
        return np.nan




if __name__ == '__main__':

    styles = pd.read_csv("./datasets/styles.csv", delimiter=',', on_bad_lines='skip')
    apparels = styles[styles["masterCategory"] == "Apparel"]
    apparels = apparels[np.logical_and(apparels["gender"] != "Boys", apparels["gender"] != "Girls")]
    apparels.drop(["masterCategory", "subCategory", "year", "productDisplayName", ], axis=1, inplace=True)
    images = os.listdir("images")
    apparels['image'] = apparels["id"].apply(lambda image: check_image_exists(str(image) + ".jpg"))
    apparels.dropna(inplace=True)

    df = apparels
    label_encoders, num_classes = prepare_data(df)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    model_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(model_name)

    train_dataset = FashionDataset(train_df, "./datasets/images", processor, label_encoders)
    val_dataset = FashionDataset(val_df, "./datasets/images", processor, label_encoders)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskViT(model_name, num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    task_weights = {
            'gender': 1.0,
            'articleType': 1.5,
            'baseColour': 1.0,
            'season': 1.0,
            'usage': 1.0
        }

    num_epochs = 10
    best_avg_accuracy = 0

    for epoch in range(num_epochs):
      print(f"\nEpoch {epoch+1}/{num_epochs}")
      train_loss = train_epoch(model, train_loader, optimizer, device, task_weights)
      print(f"Training Loss: {train_loss:.4f}")
      accuracies = evaluate(model, val_loader, device)
      avg_accuracy = np.mean(list(accuracies.values()))

      print(f"Validation Accuracies:")
      for task, acc in accuracies.items():
          print(f"  {task}: {acc:.4f}")
          print(f"  Average: {avg_accuracy:.4f}")
      if avg_accuracy > best_avg_accuracy:
          best_avg_accuracy = avg_accuracy
          torch.save({
                    'model_state_dict': model.state_dict(),
                    'label_encoders': label_encoders,
                    'num_classes': num_classes
                }, 'best_multitask_vit.pth')
          print("Model saved!")