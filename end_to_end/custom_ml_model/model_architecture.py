"""
MultiTaskViT model architecture definition.
This file contains only the model class without training code.
"""

import torch
from torch import nn
from transformers import ViTModel


class MultiTaskViT(nn.Module):
    """
    Multi-task Vision Transformer for fashion attribute prediction.
    
    Predicts 5 attributes simultaneously:
    - gender (Male/Female)
    - articleType (specific clothing type)
    - baseColour (primary color)
    - season (seasonal appropriateness)
    - usage (occasion type)
    """
    
    def __init__(self, model_name, num_classes_dict):
        super(MultiTaskViT, self).__init__()

        self.vit = ViTModel.from_pretrained(model_name)
        hidden_size = self.vit.config.hidden_size

        self.gender_classifier = nn.Linear(hidden_size, num_classes_dict['gender'])
        self.articleType_classifier = nn.Linear(hidden_size, num_classes_dict['articleType'])
        self.baseColour_classifier = nn.Linear(hidden_size, num_classes_dict['baseColour'])
        self.season_classifier = nn.Linear(hidden_size, num_classes_dict['season'])
        self.usage_classifier = nn.Linear(hidden_size, num_classes_dict['usage'])

        self.dropout = nn.Dropout(0.1)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)

        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)

        gender_logits = self.gender_classifier(cls_output)
        articleType_logits = self.articleType_classifier(cls_output)
        baseColour_logits = self.baseColour_classifier(cls_output)
        season_logits = self.season_classifier(cls_output)
        usage_logits = self.usage_classifier(cls_output)

        return {
            'gender': gender_logits,
            'articleType': articleType_logits,
            'baseColour': baseColour_logits,
            'season': season_logits,
            'usage': usage_logits
        }
