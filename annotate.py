import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import spacy
import torch
from transformers import BertTokenizer

from utils.isospace import Tag, Dimensionality, Form, SemanticType, MotionType, \
    MotionClass
from utils.model import load_model


def annotate(model, input_ids):
    with torch.no_grad():
        output = model(input_ids)
    return np.argmax(output[0].cpu().numpy(), axis=2).flatten()


def format_attribute(attribute, label):
    value = attribute.decode(label)
    return ("{}: {}".format(attribute.name, value)
            if label not in [attribute.none, attribute.pad]
            else False)


device = torch.device("cpu")

tag_model = load_model(Tag)
dimensionality_model = load_model(Dimensionality)
form_model = load_model(Form)
semantic_type_model = load_model(SemanticType)
motion_type_model = load_model(MotionType)
motion_class_model = load_model(MotionClass)

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

    tas = annotate(tag_model, ids)
    dis = annotate(dimensionality_model, ids)
    fos = annotate(form_model, ids)
    sts = annotate(semantic_type_model, ids)
    mts = annotate(motion_type_model, ids)
    mcs = annotate(motion_class_model, ids)
    words = []
    annotations = []
    for token, ta, di, fo, st, mt, mc in zip(tokens, tas, dis, fos, sts, mts,
                                             mcs):
        if token.startswith("##"):
            words[-1] = words[-1] + token[2:]
        else:
            words.append(token)
            annotations.append(", ".join([attribute for attribute in [
                format_attribute(Tag, ta),
                format_attribute(Dimensionality, di),
                format_attribute(Form, fo),
                format_attribute(SemanticType, st),
                format_attribute(MotionType, mt),
                format_attribute(MotionClass, mc),
            ] if attribute]))
    for word, annotation in zip(words, annotations):
        print("{:16s} {}".format(word, annotation))
    print()
