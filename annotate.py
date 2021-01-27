import numpy as np
import spacy
import torch
from transformers import BertTokenizer

from utils.isospace import Tag, Dimensionality, Form, SemanticType, MotionType, \
    MotionClass
from utils.model import Trainer


def annotate(attribute, input_ids):
    trainer = Trainer(attribute, device)
    with torch.no_grad():
        output = trainer.model(input_ids)
    return np.argmax(output[0].cpu().numpy(), axis=2).flatten()


device = torch.device("cpu")

sentence = "He is running towards to other wall"

nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                          do_lower_case=False)

words = [word for word in nlp(sentence) if
         word.pos_ not in ["SPACE", "PUNCT", "SYM", "."]]

ids = torch.tensor([tokenizer.encode(sentence)])
tokens = tokenizer.convert_ids_to_tokens(ids.numpy().flatten())

tags = annotate(Tag, ids)
dimensionalities = annotate(Dimensionality, ids)
forms = annotate(Form, ids)
semantic_types = annotate(SemanticType, ids)
motion_types = annotate(MotionType, ids)
motion_classes = annotate(MotionClass, ids)

row = "{:<16}" * 7
print(row.format("Word", "Tag", "Dimensionality", "Form", "SemanticType",
                 "MotionType", "MotionClass"))
print("=" * (16 * 7))
for token, tag, dimensionality, form, semantic_type, motion_type, motion_class in zip(
        tokens,
        tags,
        dimensionalities,
        forms,
        semantic_types,
        motion_classes,
        motion_types,
):
    print(row.format(
        token,
        Tag.decode(tag),
        Dimensionality.decode(dimensionality),
        Form.decode(form),
        SemanticType.decode(semantic_type),
        MotionType.decode(motion_type),
        MotionClass.decode(motion_class),
    ))
