import torch
import numpy as np
from transformers import BertTokenizer

import utils.isospace as iso
from utils.model import BertForIsoSpaceClassification
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased',
    do_lower_case=False
)

ATTRIBUTE_RULES = {
    iso.Dimensionality.id: ["PLACE", "PATH", "SPATIAL_ENTITY"],
    iso.Form.id: ["PLACE", "PATH", "SPATIAL_ENTITY"],
    iso.SemanticType.id: ["SPATIAL_SIGNAL"],
    iso.MotionType.id: ["MOTION"],
    iso.MotionClass.id: ["MOTION"],
}


class AnnotatedWord:

    def __init__(self, token, spatial_element):
        self.text = token
        self.spatial_element = iso.SpatialElement.decode(spatial_element)
        self.attributes = {}

    def add_attribute(self, attribute, value):
        if self.spatial_element in ATTRIBUTE_RULES[attribute.id]:
            self.attributes[attribute.id] = attribute.decode(value)

    def __str__(self):
        spatial_element = (
            "" if self.spatial_element == "NONE"
            else self.spatial_element)
        attributes = " | ".join([
            f"{attribute}: {value}"
            for attribute, value
            in self.attributes.items()])
        return f"{self.text:20}{spatial_element:16} {attributes}"

    def merge(self, word):
        self.text += word.text[2:]


class AnnotatedText:

    def __init__(self, sentences):
        self.sentences = sentences

    def __str__(self):
        to_print = ""
        for sent in self.sentences:
            to_print += "\n"
            for word in sent:
                to_print += str(word) + "\n"
        return to_print


class TextAnnotator:

    def __init__(self, attributes=(
            iso.Dimensionality,
            iso.Form,
            iso.SemanticType,
            iso.MotionType,
            iso.MotionClass)):
        if attributes is None:
            attributes = [
                iso.Dimensionality,
                iso.Form
            ]
        self.attributes = attributes

        self.spatial_element_model = BertForIsoSpaceClassification(
            iso.SpatialElement)
        self.attribute_models = {
            attribute.id: BertForIsoSpaceClassification(attribute) for
            attribute in attributes}

        self.spatial_element_model.load()
        for model in self.attribute_models.values():
            model.load()

    def annotate(self, text):
        sentences = tokenize_text(text)
        annotated_sentences = []
        for sent in tqdm(sentences, desc="Annotating text"):

            input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(sent)])
            with torch.no_grad():
                logits = self.spatial_element_model(input_ids)
            spatial_elements = classify(logits)

            words = [
                AnnotatedWord(token, spatial_element)
                for token, spatial_element
                in zip(sent, spatial_elements)]

            for attribute in self.attributes:
                model = self.attribute_models[attribute.id]
                with torch.no_grad():
                    logits = model(input_ids)
                values = classify(logits)
                for word, value in zip(words, values):
                    word.add_attribute(attribute, value)

            merged_words = []
            for word in words:
                if word.text.startswith("##"):
                    merged_words[-1].merge(word)
                else:
                    merged_words.append(word)
            annotated_sentences.append(merged_words)
        return AnnotatedText(annotated_sentences)


def tokenize_text(text):
    sentences = []
    for sent in nlp(text).sents:
        if len(sent) > 0:
            tokens = tokenizer.tokenize(sent.text)
            if len(tokens) > 0:
                sentences.append(tokens)
    return sentences


def classify(logits):
    return np.argmax(logits.detach().numpy(), axis=2).flatten()
