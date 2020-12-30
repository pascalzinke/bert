import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import glob
import os
import spacy
from transformers import BertTokenizer
import torch
from tqdm import tqdm
from time import sleep
import numpy as np

MAX_LENGTH = 50  # max length of sentence
MAKE_DATA = False  # start variable for the annotated dataset

data_path = os.path.join("data", "data.npy")

nlp = spacy.load("en_core_web_sm")  # spacy english model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)   #standard bert tokenizer

"""
This class is used to:
1.) Map all spatial entity types to an label
2.) Get attributes from the tags 
"""


class IsoSpaceEntity:
    # types of IsoSpace
    __types = ["PAD", "NONE", "PLACE", "PATH", "SPATIAL_ENTITY", "NONMOTION_EVENT", "MOTION", "SPATIAL_SIGNAL",
               "MOTION_SIGNAL", "MEASURE"]

    # mapping IsoSpace types to integer for bert model
    __tag2label = {t: i for i, t in enumerate(__types)}

    def __init__(self, tag):
        self.label = self.tag_to_label(tag.tag) if tag.tag in IsoSpaceEntity.__types else 0  #labeling including PAD
        self.id = tag.get("id")
        self.text = tag.get("text")
        start, end = tag.get("start"), tag.get("end")
        self.start = int(start) if start else None
        self.end = int(end) if end else None
        self.interval = range(self.start, self.end) if self.start and self.end else []

    @staticmethod
    def tag_to_label(tag):
        return next(label for label, t in enumerate(IsoSpaceEntity.__types) if tag == t)

    @staticmethod
    def label_to_tag(label):
        return IsoSpaceEntity.__types[label]

    @staticmethod
    def n_types():
        return len(IsoSpaceEntity.__types)


"""
This class is used to preprocess XML files in Data for Bert
Input: XML files ( Sentences and Tags)
Output: torch.tensor([token_ids, labels, attention_mask]) 
"""


class AnnotatedTextDataset(Dataset):

    def __init__(self):
        if MAKE_DATA:
            sentences = []  # list of sentences
            xml_files = glob.glob(os.path.join("data", "**", "*.xml"), recursive=True)
            print("\nExtracting xml files...")
            sleep(0.1)
            for xml_file in tqdm(xml_files):  # going through xml files
                tree = ET.parse(xml_file)
                iso_space_entities = [IsoSpaceEntity(tag) for tag in tree.find("TAGS")]  # creating object for tags
                text = tree.find("TEXT").text  # find the text for the object
                for sentence in nlp(text).sents:
                    current_sentence = []
                    for word in sentence:
                        if word.pos_ not in ["SPACE", "PUNCT", "SYM", "."]:
                            label = next((ise.label for ise in iso_space_entities if word.idx in ise.interval), 1)
                            tokens = tokenizer.tokenize(word.text)  # tokenizing word
                            token_ids = tokenizer.convert_tokens_to_ids(tokens)  # giving id to token
                            current_sentence.extend([(token_id, label) for token_id in token_ids])  # add to sentence
                    length = len(current_sentence)
                    if length > MAX_LENGTH:
                        current_sentence = current_sentence[:MAX_LENGTH]
                    else:  # padding sentence to max_length
                        current_sentence += [(0, 0)] * (MAX_LENGTH - length)
                    sentences.append(current_sentence)
            print("\n")
            self.sentences = np.array(sentences)
            np.save(data_path, self.sentences)  # save data path with annotated sentences
        else:
            self.sentences = np.load(data_path)  # if data was already processed
        pass

    def __getitem__(self, idx):  # getter for sentence
        sentence = self.sentences[idx]
        token_ids = [token_id for token_id, label in sentence]
        labels = [label for token_id, label in sentence]
        attention_mask = [int(label != 0) for token_id, label in sentence]
        return torch.tensor([token_ids, labels, attention_mask])  # return as tensor

    def __len__(self):
        return len(self.sentences)
