import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import glob
import os
import spacy
from transformers import BertTokenizer
import torch

MAX_LENGTH = 50

nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)


class IsoSpaceEntity:
    __types = ["NONE", "PAD", "PLACE", "PATH", "SPATIAL_ENTITY", "NONMOTION_EVENT", "MOTION", "SPATIAL_SIGNAL",
               "MOTION_SIGNAL", "MEASURE"]

    __tag2label = {t: i for i, t in enumerate(__types)}

    def __init__(self, tag):
        self.label = self.tag_to_label(tag.tag) if tag.tag in IsoSpaceEntity.__types else 0
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


class AnnotatedTextDataset(Dataset):

    def __init__(self):
        self.sentences = []
        xml_files = glob.glob(os.path.join("data", "**", "*.xml"), recursive=True)
        longer = 0
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            iso_space_entities = [IsoSpaceEntity(tag) for tag in tree.find("TAGS")]
            text = tree.find("TEXT").text
            for sentence in nlp(text).sents:
                current_sentence = []
                for word in sentence:
                    if word.pos_ not in ["SPACE", "PUNCT", "SYM", "."]:
                        label = next((ise.label for ise in iso_space_entities if word.idx in ise.interval), 0)
                        tokens = tokenizer.tokenize(word.text)
                        token_ids = tokenizer.convert_tokens_to_ids(tokens)
                        current_sentence.extend([(token_id, label) for token_id in token_ids])
                length = len(current_sentence)
                if length > MAX_LENGTH:
                    current_sentence = current_sentence[:MAX_LENGTH]
                else:
                    current_sentence += [(0, 1)] * (MAX_LENGTH - length)
                self.sentences.append(current_sentence)
        pass

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        token_ids = [token_id for token_id, label in sentence]
        labels = [label for token_id, label in sentence]
        return torch.tensor([token_ids, labels])

    def __len__(self):
        return len(self.sentences)
