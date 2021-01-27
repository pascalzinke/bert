import os

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import BertForTokenClassification, AdamW, \
    get_linear_schedule_with_warmup


class Trainer:

    def __init__(self, attribute, device, train_loader, fresh=False, epochs=3):
        self.attribute = attribute
        self.device = device
        self.path = os.path.join("cache",
                                 "{}_model.pt".format(self.attribute.name))
        self.model = BertForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=self.attribute.n,
            output_attentions=False,
            output_hidden_states=False
        )
        if os.path.isfile(self.path) and not fresh:
            print("Loading {} model...".format(self.attribute.name))
            self.model.load_state_dict(torch.load(self.path))
            print("Model loaded\n")
        self.model.to(self.device)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if
                        not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if
                        any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=3e-5,
            eps=1e-8
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader) * epochs
        )
        pass

    def save(self):
        print("Saving {} model...".format(self.attribute.name))
        torch.save(self.model.state_dict(), self.path)
        print("Model saved\n")
        pass

    def train(self, b_ids, b_labels, b_mask):
        self.model.train()
        self.model.zero_grad()
        outputs = self.model(b_ids, labels=b_labels, attention_mask=b_mask)
        loss = outputs[0]
        loss.backward()
        clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def eval(self, b_ids, b_labels, b_mask):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(b_ids, token_type_ids=None,
                                 attention_mask=b_mask,
                                 labels=b_labels)
        logits = outputs[1].cpu().numpy()
        total, correct = 0, 0
        pred_labels = np.argmax(logits, axis=2).flatten()
        true_labels = b_labels.detach().cpu().numpy().flatten()
        for pred_label, true_label in zip(pred_labels, true_labels):
            if true_label != 0:
                total += 1
                if pred_label == true_label:
                    correct += 1
        return correct, total
