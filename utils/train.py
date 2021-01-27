from time import sleep

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.model import initialize_model, save_model


def train_model(attribute, train_loader, eval_loader, device, epochs=4):
    print("Training {} model".format(attribute.name))

    model = initialize_model(attribute)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        eps=1e-8
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * epochs
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc="Epoch {}".format(epoch + 1)):
            batch = tuple(t.to(device) for t in batch)
            b_ids = batch[0]
            b_labels = batch[attribute.index]
            b_mask = batch[7]

            model.zero_grad()
            outputs = model(b_ids, labels=b_labels, attention_mask=b_mask)
            loss = outputs[0]
            loss.backward()
            clip_grad_norm_(parameters=model.parameters(), max_norm=1.)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        for batch in eval_loader:
            batch = tuple(t.to(device) for t in batch)
            b_ids = batch[0]
            b_labels = batch[attribute.index]
            b_mask = batch[7]

            with torch.no_grad():
                outputs = model(b_ids, token_type_ids=None,
                                attention_mask=b_mask,
                                labels=b_labels)
            logits = outputs[1].cpu().numpy()

            pred_labels = np.argmax(logits, axis=2).flatten()
            true_labels = b_labels.detach().cpu().numpy().flatten()
            for pred_label, true_label in zip(pred_labels, true_labels):
                if true_label != 0:
                    total += 1
                    if pred_label == true_label:
                        correct += 1

        print("Loss:     {:.2%}".format(total_loss / len(train_loader)))
        print("Accuracy: {:.2%}".format(correct / total))
        sleep(0.1)

    save_model(model, attribute)
