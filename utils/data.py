import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import spacy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

import utils.isospace as iso

MAX_LENGTH = 50
NONE_WORDS = ["SPACE", "PUNCT", "SYM", "."]

nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased',
    do_lower_case=False
)


def extract_data(data_type):
    sentences = []

    # Read all xml files
    xml_files = glob.glob(
        os.path.join("data", data_type, "**", "*.xml"),
        recursive=True
    )
    desc = "Extracting {} data".format(data_type)
    for xml_file in tqdm(xml_files, desc=desc):
        tree = ET.parse(xml_file)

        # Initialize iso space helper entities from annotated xml tags
        entities = [iso.IsoSpaceElement(tag) for tag in tree.find("TAGS")]

        # Tokenize test with spacy
        text = tree.find("TEXT").text
        for sentence in nlp(text).sents:
            current_sentence = []
            for word in sentence:

                # Filter none words
                if word.pos_ not in NONE_WORDS:
                    # Tokenize with bert tokenizer
                    tokens = tokenizer.tokenize(word.text)
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)

                    # Find corresponding iso space entity
                    entity = next(
                        (e for e in entities if
                         word.idx in e.interval), iso.IsoSpaceElement(None))

                    # Append token, spatial element type, and attribute values
                    # to sentence
                    current_sentence.extend(
                        [[
                            token_id,
                            entity.spatial_element,
                            entity.dimensionality,
                            entity.form,
                            entity.semantic_type,
                            entity.motion_type,
                            entity.motion_class,
                        ] for token_id in token_ids]
                    )

            # Pad/cut sentences to a constant size
            sentences.append(pad_sentence(current_sentence, [
                0,
                iso.SpatialElement.pad,
                iso.Dimensionality.pad,
                iso.Form.pad,
                iso.SemanticType.pad,
                iso.MotionType.pad,
                iso.MotionClass.pad,
            ]))

    # Cache extracted data
    data = np.array(sentences)
    np.save(get_data_path_name(data_type), data)
    return data


def get_data_path_name(data_type):
    return os.path.join("cache", "{}.npy".format(data_type))


def pad_sentence(sentence, padding):
    # Pad or cut sentence to a constant size
    length = len(sentence)
    return (
        sentence[:MAX_LENGTH]
        if length > MAX_LENGTH
        else sentence + [padding] * (MAX_LENGTH - length))


class TextDataset(Dataset):

    def __init__(self, evaluate=False):
        # Load cached data if available else extract data from xml files
        data_type = "eval" if evaluate else "train"
        path = get_data_path_name(data_type)
        self.sentences = (
            np.load(path)
            if os.path.isfile(path)
            else extract_data(data_type))
        pass

    def __getitem__(self, idx):
        # Returns token, spatial element type, and attribute values as a
        # dictionary. Needed for pytorch DataLoader
        # Creates a mask for padded sentences
        sentence = self.sentences[idx]
        ids = torch.tensor(sentence[:, 0], dtype=torch.long)
        return {
            "token_ids": ids,
            "mask": torch.tensor([int(i != 0) for i in ids], dtype=torch.long),
            iso.SpatialElement.id: torch.tensor(sentence[:, 1], dtype=torch.long),
            iso.Dimensionality.id: torch.tensor(sentence[:, 2], dtype=torch.long),
            iso.Form.id: torch.tensor(sentence[:, 3], dtype=torch.long),
            iso.SemanticType.id: torch.tensor(sentence[:, 4], dtype=torch.long),
            iso.MotionType.id: torch.tensor(sentence[:, 5], dtype=torch.long),
            iso.MotionClass.id: torch.tensor(sentence[:, 6], dtype=torch.long),
        }

    def __len__(self):
        return len(self.sentences)


def get_test_data():
    # Since test data has no annotations, extraction is much simpler
    text = ""
    xml_files = glob.glob(
        os.path.join("data", "test", "**", "*.xml"),
        recursive=True)
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        text += tree.find("TEXT").text
    return text
