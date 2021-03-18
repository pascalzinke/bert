import torch
import numpy as np
from transformers import BertTokenizer

import utils.isospace as iso
from utils.model import BertForIsoSpaceClassification
import spacy
from tqdm import tqdm

# Load spacy and bert tokenizers
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased',
    do_lower_case=False
)

# Which spatial elements can have which attributes
ATTRIBUTE_RULES = {
    iso.Dimensionality.id: ["PLACE", "PATH", "SPATIAL_ENTITY"],
    iso.Form.id: ["PLACE", "PATH", "SPATIAL_ENTITY"],
    iso.SemanticType.id: ["SPATIAL_SIGNAL"],
    iso.MotionType.id: ["MOTION"],
    iso.MotionClass.id: ["MOTION"],
}


class AnnotatedWord:
    """
    Helper class for storing the text, spatial element type, and attributes of
    words
    """

    def __init__(self, text, spatial_element):
        # Stores the text and the encoded spatial element type of a word
        self.text = text
        self.spatial_element = iso.SpatialElement.decode(spatial_element)
        self.attributes = {}

    def add_attribute(self, attribute, value):
        # Adds an attribute to the word, if given attributes exists on the
        # spatial element type
        if self.spatial_element in ATTRIBUTE_RULES[attribute.id]:
            self.attributes[attribute.id] = attribute.decode(value)

    def __str__(self):
        # Format word for printing annotated texts
        spatial_element = (
            "" if self.spatial_element == "NONE"
            else self.spatial_element)
        attributes = " | ".join([
            f"{attribute}: {value}"
            for attribute, value
            in self.attributes.items()])
        return f"{self.text:20}{spatial_element:16} {attributes}"


class AnnotatedText:
    """
    Stores annotated sentences. Currently only used for printing annotated text.
    Can be extended for exporting or saving annotated data
    """

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
    """
    Used for annotating a given text
    """

    def __init__(self, attributes=(
            iso.Dimensionality,
            iso.Form,
            iso.SemanticType,
            iso.MotionType,
            iso.MotionClass)):
        # Takes a list of attributes to be annotated. Spatial Element types are
        # always annotated and do not need to be specified
        self.attributes = attributes

        # Initialize spatial element and attribute models
        self.spatial_element_model = BertForIsoSpaceClassification(
            iso.SpatialElement)
        self.attribute_models = {
            attribute.id: BertForIsoSpaceClassification(attribute) for
            attribute in attributes}

        # Load trained models
        self.spatial_element_model.load()
        for model in self.attribute_models.values():
            model.load()

    def annotate(self, text):
        # Split sentences with spacy and tokenize sentences with bert tokenizer
        sentences = []
        for sent in nlp(text).sents:
            if len(sent) > 0:
                tokens = tokenizer.tokenize(sent.text)
                if len(tokens) > 0:
                    sentences.append(tokens)

        annotated_sentences = []
        for sent in tqdm(sentences, desc="Annotating text"):

            # Create spatial element type annotations
            input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(sent)])
            with torch.no_grad():
                logits = self.spatial_element_model(input_ids)
            spatial_elements = np.argmax(
                logits.detach().numpy(), axis=2).flatten()

            # Instantiate helper class for each word with spatial element type
            words = [
                AnnotatedWord(token, spatial_element)
                for token, spatial_element
                in zip(sent, spatial_elements)]

            # Create attribute annotation for each attribute specified
            for attribute in self.attributes:
                model = self.attribute_models[attribute.id]
                with torch.no_grad():
                    logits = model(input_ids)
                values = np.argmax(logits.detach().numpy(), axis=2).flatten()

                # Add attribute annotation to each word.
                for word, value in zip(words, values):
                    word.add_attribute(attribute, value)

            # Since the bert tokenizer splits long words into syllables, those
            # syllables need to be merged again.
            # Trailing syllables start with "##"
            merged_words = []
            for word in words:
                if word.text.startswith("##"):
                    merged_words[-1].text += word.text[2:]
                else:
                    merged_words.append(word)
            annotated_sentences.append(merged_words)

        # Return helper class with all annotated sentences
        return AnnotatedText(annotated_sentences)

