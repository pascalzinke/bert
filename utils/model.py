import os

import torch
import torch.nn as nn
from transformers import BertModel
import warnings

BERT_MODEL = "bert-base-cased"
BERT_LAST_LAYER = 768


class BertForAttributeClassification(nn.Module):

    def __init__(self, attribute):
        super().__init__()
        self.model_path = os.path.join("cache", f"{attribute.id}_model.pt")

        self.bert = BertModel.from_pretrained(
            BERT_MODEL, add_pooling_layer=False)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            BERT_LAST_LAYER, attribute.num_values)

    def forward(self, input_ids, mask=None):
        outputs = self.bert(input_ids, attention_mask=mask)[0]

        x = self.dropout(outputs)
        logits = self.classifier(x)

        return logits

    def save(self):
        print("Saving model...")
        torch.save(self.state_dict(), self.model_path)
        print("Model saved")

    def load(self):
        if os.path.isfile(self.model_path):
            print("Loading model...")
            self.load_state_dict(torch.load(self.model_path))
            print("Model loaded")
        else:
            warnings.warn(f"No model found in {self.model_path}")

