import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import spacy
import torch
from transformers import BertTokenizer

from utils.isospace import SpatialElement, Dimensionality, Form, SemanticType, \
    MotionType, MotionClass
from utils.model import BertForSpatialElementClassification

model = BertForSpatialElementClassification()
model.load()

device = torch.device("cpu")

nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                          do_lower_case=False)

text = ""
xml_files = glob.glob(
    os.path.join("data", "test", "**", "*.xml"),
    recursive=True
)
for xml_file in xml_files:
    tree = ET.parse(xml_file)
    text += tree.find("TEXT").text

sentences = [[word.text for word in sentence if
              word.pos_ not in ["SPACE", "PUNCT", "SYM", "."]] for sentence in
             nlp(text).sents if len(sentence) > 0]

for sentence in sentences:
    tokens = tokenizer.tokenize(" ".join(sentence))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    ids = torch.tensor([token_ids], dtype=torch.long)

    spatial_element_logits, dimensionality_logits = model(ids)

    spatial_elements = np.argmax(spatial_element_logits.detach().numpy(), axis=2).flatten()
    dimensionalities = np.argmax(dimensionality_logits.detach().numpy(), axis=2).flatten()
    words = []
    annotations = []
    for token, spatial_element, dimensionality in zip(tokens, spatial_elements,
                                                      dimensionalities):
        if token.startswith("##"):
            words[-1] = words[-1] + token[2:]
        else:
            words.append(token)
            annotation = ""
            if spatial_element != SpatialElement.none:
                annotation += SpatialElement.decode(spatial_element)
            if dimensionality != Dimensionality.none:
                annotation += Dimensionality.decode(dimensionality)
            annotations.append(annotation)
    for word, annotation in zip(words, annotations):
        print("{:16s} {}".format(word, annotation))
    print()
