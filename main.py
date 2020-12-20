from preprocess import AnnotatedTextDataset, IsoSpaceEntity
from torch.utils.data import DataLoader, random_split
import torch
from transformers import BertForTokenClassification, AdamW

BATCH_SIZE = 10
EPOCHS = 3

dataset = AnnotatedTextDataset()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on " + torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Running on CPU")

model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=IsoSpaceEntity.n_types(),
    output_attentions=False,
    output_hidden_states=False
)
