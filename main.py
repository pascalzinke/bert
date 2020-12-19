from preprocess.data import AnnotatedTextDataset
from torch.utils.data import DataLoader, random_split

BATCH_SIZE = 10

dataset = AnnotatedTextDataset()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_set, test_set = random_split(dataset, [train_size, test_size])

print(len(train_set))
print(len(test_set))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
