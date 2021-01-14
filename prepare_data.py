import torch
from torch.utils.data import DataLoader, random_split

from preprocess import AnnotatedTextDataset
from constants import BATCH_SIZE, TRAIN_LOADER_PATH, TEST_LOADER_PATH

dataset = AnnotatedTextDataset()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

torch.manual_seed(0)
train_set, test_set = random_split(dataset, [train_size, test_size])
torch.manual_seed(torch.initial_seed())

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

torch.save(train_loader, TRAIN_LOADER_PATH)
torch.save(test_loader, TEST_LOADER_PATH)
