import os

import torch
from transformers import BertForTokenClassification


def get_model_path(attribute):
    return os.path.join("cache", "{}_model.pt".format(attribute.name))


def initialize_model(attribute):
    return BertForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=attribute.n,
        output_attentions=False,
        output_hidden_states=False
    )


def save_model(model, attribute):
    print("Saving {} model...".format(attribute.name))
    path = get_model_path(attribute)
    torch.save(model.state_dict(), path)
    print("Model saved\n")


def load_model(attribute):
    model = initialize_model(attribute)
    path = get_model_path(attribute)
    if os.path.isfile(path):
        print("Loading {} model...".format(attribute.name))
        model.load_state_dict(torch.load(path))
        print("Model loaded\n")
    return model
