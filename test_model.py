import torch
from transformers import BertForTokenClassification
import numpy as np

from constants import MODEL_PATH, TEST_LOADER_PATH
from preprocess import IsoSpaceEntity

test_loader = torch.load(TEST_LOADER_PATH)

print("Loading model...")
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=IsoSpaceEntity.n_types(),
    output_attentions=False,
    output_hidden_states=False
)
model.load_state_dict(torch.load(MODEL_PATH))

model.eval()
pred_labels, true_labels = [], []
total_words, correct_words = 0, 0
for batch in test_loader:
    token_ids, labels, attention_mask = batch

    with torch.no_grad():
        outputs = model(token_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels)
    logits = outputs[1].numpy()
    pred_labels.extend([list(p) for p in np.argmax(logits, axis=2)])
    true_labels.extend(labels.numpy())

for pred_sentence, true_sentence in zip(pred_labels, true_labels):
    for pred_label, true_label in zip(pred_sentence, true_sentence):
        if true_label != IsoSpaceEntity.tag_to_label("PAD"):
            total_words += 1
            if true_label == pred_label:
                correct_words += 1

print("Accuracy: {}%".format(correct_words / total_words))

