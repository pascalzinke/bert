import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup

from preprocess import IsoSpaceEntity
from constants import EPOCHS, MODEL_PATH, TRAIN_LOADER_PATH

train_loader = torch.load(TRAIN_LOADER_PATH)

model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=IsoSpaceEntity.n_types(),
    output_attentions=False,
    output_hidden_states=False
)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on " + torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Running on CPU")
print()

model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(train_loader) * EPOCHS
)
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
