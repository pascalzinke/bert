from preprocess import AnnotatedTextDataset, IsoSpaceEntity
from torch.utils.data import DataLoader, random_split
import torch
from transformers import BertForTokenClassification, AdamW, BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import numpy as np

BATCH_SIZE = 10
EPOCHS = 3
MODEL_PATH = "model.pt"
TRAIN_MODEL = True

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
dataset = AnnotatedTextDataset(tokenizer)

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

torch.manual_seed(0)
train_set, test_set = random_split(dataset, [train_size, test_size])
torch.manual_seed(torch.initial_seed())

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on " + torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Running on CPU")
print()

model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=IsoSpaceEntity.n_types(),
    output_attentions=False,
    output_hidden_states=False
)
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
]
optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * EPOCHS
)

if TRAIN_MODEL:
    for epoch in range(EPOCHS):
        print("Epoch {}:".format(epoch + 1))
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            batch = tuple(t.to(device) for t in batch)
            token_ids, labels, attention_mask = batch
            model.zero_grad()
            outputs = model(token_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()
            clip_grad_norm_(parameters=model.parameters(), max_norm=1.)
            optimizer.step()
            scheduler.step()
        train_loss = total_loss / len(train_loader)
        print("Loss: {}\n".format(train_loss))
    torch.save(model.state_dict(), MODEL_PATH)
else:
    model.load_state_dict(torch.load(MODEL_PATH))

model.eval()
with torch.no_grad():
    batch = tuple(t.to(device) for t in list(test_loader)[0])
    token_ids, labels, attention_mask = batch
    output = model(token_ids, labels=labels, attention_mask=attention_mask)
    words = [tokenizer.convert_ids_to_tokens(batch) for batch in token_ids]
    tags = [[IsoSpaceEntity.label_to_tag(label) for label in batch] for batch in labels]
    pred_labels = np.argmax(output.logits.detach().cpu().numpy(), axis=2)
    for batch in np.dstack((words, pred_labels, tags)):
        for word, label, tag in batch:
            if word != "[PAD]":
                print(word, IsoSpaceEntity.label_to_tag(int(label)), tag)
