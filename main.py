from preprocess import AnnotatedTextDataset, IsoSpaceEntity
from torch.utils.data import DataLoader, random_split
import torch
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import trange
from torch.nn.utils import clip_grad_norm_

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

optimizer = AdamW(
    [{"params": [p for n, p in list(model.classifier.named_parameters())]}],
    lr=3e-5,
    eps=1e-8
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * EPOCHS
)

for _ in trange(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        token_ids = batch[:, 0, :]
        labels = batch[:, 1, :]
        attention_mask = batch[:, 2, :]
        model.zero_grad()
        outputs = model(token_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs[0]
        loss.backward()
        total_loss += loss.item()
        clip_grad_norm_(parameters=model.parameters(), max_norm=1.)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_loader)
    print("Average train loss: {}".format(avg_train_loss))

