import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import spacy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from utils.isospace import Entity, SpatialElement, Dimensionality, Form, SemanticType, \
    MotionType, MotionClass

MAX_LENGTH = 50

nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased',
    do_lower_case=False
)


def extract_data(data_type):
    sentences = []

    # read all xml files
    xml_files = glob.glob(
        os.path.join("data", data_type, "**", "*.xml"),
        recursive=True
    )
    desc = "Extracting {} data".format(data_type)
    for xml_file in tqdm(xml_files, desc=desc):
        tree = ET.parse(xml_file)

        # initialize iso space helper entities from annotated xml tags
        entities = [Entity(tag) for tag in tree.find("TAGS")]

        # tokenize test with spacy
        text = tree.find("TEXT").text
        for sentence in nlp(text).sents:
            current_sentence = []
            for word in sentence:

                # filter non words
                if word.pos_ not in ["SPACE", "PUNCT", "SYM", "."]:
                    # find corresponding iso space entity
                    entity = next(
                        (e for e in entities if
                         word.idx in e.interval), Entity(None))

                    # create embeddings
                    tokens = tokenizer.tokenize(word.text)
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)

                    # append token-label pair to sentence
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

            # pad/cut sentences to constant size
            sentences.append(pad_sentence(current_sentence, [
                0,
                SpatialElement.pad,
                Dimensionality.pad,
                Form.pad,
                SemanticType.pad,
                MotionType.pad,
                MotionClass.pad,
            ]))

    # save extracted data to file
    data = np.array(sentences)
    np.save(get_data_path_name(data_type), data)
    return data


def get_data_path_name(data_type):
    return os.path.join("cache", "{}.npy".format(data_type))


def pad_sentence(sentence, padding):
    # pad or cut sentence to constant size
    length = len(sentence)
    return (
        sentence[:MAX_LENGTH]
        if length > MAX_LENGTH
        else sentence + [padding] * (MAX_LENGTH - length))


class AnnotatedDataset(Dataset):

    def __init__(self, data_type):
        # load annotated data if available else extract data from xml files
        path = get_data_path_name(data_type)
        self.sentences = (
            np.load(path)
            if os.path.isfile(path)
            else extract_data(data_type))
        pass

    def __getitem__(self, idx):
        # get sentence
        sentence = self.sentences[idx]
        ids = torch.tensor(sentence[:, 0], dtype=torch.long)
        return {
            "token_ids": ids,
            "mask": torch.tensor([int(i != 0) for i in ids], dtype=torch.long),
            SpatialElement.id: torch.tensor(sentence[:, 1], dtype=torch.long),
            Dimensionality.id: torch.tensor(sentence[:, 2], dtype=torch.long),
            Form.id: torch.tensor(sentence[:, 3], dtype=torch.long),
            SemanticType.id: torch.tensor(sentence[:, 4], dtype=torch.long),
            MotionType.id: torch.tensor(sentence[:, 5], dtype=torch.long),
            MotionClass.id: torch.tensor(sentence[:, 6], dtype=torch.long),
        }

    def __len__(self):
        # get length of dataset

        return len(self.sentences)

