import torch
from transformers import BertForTokenClassification

from constants import MODEL_PATH
from preprocess import IsoSpaceEntity

model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=IsoSpaceEntity.n_types(),
    output_attentions=False,
    output_hidden_states=False
)

model.load_state_dict(torch.load(MODEL_PATH))

sentence = "Inside the room are three windows and next to the door is a big painting on the wall"


