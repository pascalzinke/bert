from transformers import BertForTokenClassification
from utils.isospace import IsoSpaceEntity
from utils.data import AnnotatedDataset
import torch
from torch.utils.data import DataLoader
import os
import numpy as np

MODEL_PATH = os.path.join("cache", "model.pt")


def save_model(model):
    torch.save(model.state_dict(), MODEL_PATH)


def load_model():
    print("Loading model...")
    model = BertForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=IsoSpaceEntity.n_types(),
        output_attentions=False,
        output_hidden_states=False
    )
    if os.path.isfile(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
    return model


def evaluate_model(model, device):
    dataset = AnnotatedDataset("eval")
    loader = DataLoader(dataset, batch_size=10)

    model.eval()
    pred_batch, true_batch = [], []
    total_words, correct_words = 0, 0
    for batch in loader:
        batch = tuple(t.to(device) for t in batch)
        token_ids_batch, labels_batch, attention_mask_batch = batch

        with torch.no_grad():
            outputs = model(token_ids_batch, token_type_ids=None, attention_mask=attention_mask_batch,
                            labels=labels_batch)
        logits = outputs[1].detach().cpu().numpy()
        pred_batch.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_batch.extend(labels_batch.detach().cpu().numpy())

    for pred_sentence, true_sentence in zip(pred_batch, true_batch):
        for pred_label, true_label in zip(pred_sentence, true_sentence):
            if not IsoSpaceEntity.is_padding(true_label):
                total_words += 1
                if true_label == pred_label:
                    correct_words += 1

    accuracy = correct_words / total_words
    return accuracy
