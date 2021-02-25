import os

import torch
import torch.nn as nn
from transformers import BertModel
from utils.isospace import SpatialElement, Dimensionality, Form, SemanticType, \
    MotionType, MotionClass
import warnings

BERT_MODEL = "bert-base-cased"
BERT_LAST_LAYER = 768
MODEL_PATH = os.path.join("cache", "model.pt")


class BertForSpatialElementClassification(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            BERT_MODEL, add_pooling_layer=False)

        self.spatial_element_fc = nn.Linear(BERT_LAST_LAYER, 512)
        self.spatial_element_classifier = nn.Linear(
            512, SpatialElement.num_values)

        self.dimensionality_fc = nn.Linear(BERT_LAST_LAYER + SpatialElement.num_values, 512)
        self.dimensionality_classifier = nn.Linear(
            512, Dimensionality.num_values)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, mask=None):
        outputs = self.bert(input_ids, attention_mask=mask)[0]

        x = self.spatial_element_fc(outputs)
        x = self.dropout(self.relu(x))
        spatial_element_logits = self.spatial_element_classifier(x)

        x = torch.cat((outputs, spatial_element_logits), 2)
        x = self.dimensionality_fc(x)
        x = self.dropout(self.relu(x))
        dimensionality_logits = self.dimensionality_classifier(x)

        return spatial_element_logits, dimensionality_logits

    def save(self):
        print("Saving model...")
        torch.save(self.state_dict(), MODEL_PATH)
        print("Model saved")

    def load(self):
        if os.path.isfile(MODEL_PATH):
            print("Loading model...")
            self.load_state_dict(torch.load(MODEL_PATH))
            print("Model loaded")
        else:
            warnings.warn(f"No model found in {MODEL_PATH}")
