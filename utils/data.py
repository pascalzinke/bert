import glob
import os
import xml.etree.ElementTree as ET

import numpy as np
import spacy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from utils.isospace import IsoSpaceEntity

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
        iso_space_entities = [IsoSpaceEntity(tag) for tag in tree.find("TAGS")]

        # tokenize test with spacy
        text = tree.find("TEXT").text
        for sentence in nlp(text).sents:
            current_sentence = []
            for word in sentence:

                # filter non words
                if word.pos_ not in ["SPACE", "PUNCT", "SYM", "."]:
                    # find corresponding iso space entity
                    label = next((ise.label for ise in iso_space_entities if
                                  word.idx in ise.interval), 1)

                    # create embeddings
                    tokens = tokenizer.tokenize(word.text)
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)

                    # append token-label pair to sentence
                    current_sentence.extend(
                        [[token_id, label] for token_id in token_ids]
                    )

            # pad/cut sentences to constant size
            sentences.append(pad_sentence(current_sentence, [0, 0]))

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
        else sentence + [padding] * (MAX_LENGTH - length)
    )


class AnnotatedDataset(Dataset):

    def __init__(self, data_type):
        # load annotated data if available else extract data from xml files
        path = get_data_path_name(data_type)
        self.sentences = (
            np.load(path)
            if os.path.isfile(path)
            else extract_data(data_type)
        )
        pass

    def __getitem__(self, idx):
        # get sentence
        sentence = self.sentences[idx]

        # get embeddings and labels for sentence
        token_ids = torch.tensor(
            [token_id for token_id, label in sentence], dtype=torch.long
        )
        labels = torch.tensor(
            [label for token_id, label in sentence], dtype=torch.long
        )

        # mask sentence padding
        attention_mask = torch.tensor(
            [int(label != 0) for label in labels], dtype=torch.long
        )
        return token_ids, labels, attention_mask

    def __len__(self):
        # get length of dataset
        return len(self.sentences)
