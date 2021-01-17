import torch
import spacy
from transformers import BertTokenizer
import numpy as np
from utils.model import load_model
from utils.isospace import IsoSpaceEntity

model = load_model()

sentence = "Inside the room are three windows and next to the door is a big painting on the wall."

nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

words = [word for word in nlp(sentence) if word.pos_ not in ["SPACE", "PUNCT", "SYM", "."]]

input_ids = torch.tensor([tokenizer.encode(sentence)])

with torch.no_grad():
    output = model(input_ids)
labels = np.argmax(output[0].numpy(), axis=2)
tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
new_tokens, new_labels = [], []
for token, label in zip(tokens, labels[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(IsoSpaceEntity.label_to_tag(label))
        new_tokens.append(token)
for token, label in zip(new_tokens, new_labels):
    print(label.ljust(20) + token)
