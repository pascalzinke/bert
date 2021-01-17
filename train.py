from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.cuda import get_device
from utils.model import load_model, evaluate_model, save_model
from utils.data import AnnotatedDataset
from torch.utils.data import DataLoader
from time import sleep

EPOCHS = 9
BATCH = 8

model = load_model()
device = get_device()
model.to(device)

dataset = AnnotatedDataset("train")
loader = DataLoader(dataset, batch_size=BATCH)

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
    num_training_steps=len(loader) * EPOCHS
)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Epoch {}".format(epoch + 1)):
        batch = tuple(t.to(device) for t in batch)
        token_ids_batch, labels_batch, attention_mask_batch = batch
        model.zero_grad()
        outputs = model(token_ids_batch, labels=labels_batch, attention_mask=attention_mask_batch)
        loss = outputs[0]
        loss.backward()
        total_loss += loss.item()
        clip_grad_norm_(parameters=model.parameters(), max_norm=1.)
        optimizer.step()
        scheduler.step()
    train_loss = total_loss / len(loader)
    accuracy = evaluate_model(model, device)
    print("Loss:     {}".format(train_loss))
    print("Accuracy: {}".format(accuracy))
    print()
    sleep(0.1)

save_model(model)
