from time import sleep

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, \
    BertForTokenClassification

import utils.isospace as iso
from utils.model import BertForAttributeClassification


class TrainerConfig:
    def __init__(self, device, train_loader, eval_loader):
        self.device = device
        self.train_loader = train_loader
        self.eval_loader = eval_loader


class AttributeTrainer:

    def __init__(self, attribute, config, epochs, lr=3e-5):
        self.model = BertForAttributeClassification(attribute)
        self.config = config
        self.attribute = attribute
        self.epochs = epochs
        self.model.to(self.config.device)

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
            optimizer_grouped_parameters, lr=lr, eps=1e-8)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.config.train_loader) * self.epochs
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def _train(self, batch):
        unpacker = Unpacker(batch, self.config.device)

        self.optimizer.zero_grad()
        logits = self.model(unpacker.token_ids, mask=unpacker.mask)
        labels = unpacker.get(self.attribute)

        active_logits = logits.view(-1, self.attribute.num_values)
        active_labels = self._mask_labels(labels).view(-1)

        loss = self.loss_fn(active_logits, active_labels)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        unpacker.detach()

        return loss.item()

    def _mask_labels(self, labels):
        mask_pad = torch.eq(labels, self.attribute.pad)
        mask_none = torch.eq(labels, self.attribute.none)
        mask = torch.logical_or(mask_pad, mask_none)
        return torch.where(mask, self.loss_fn.ignore_index, labels)

    def _eval(self, batch):
        unpacker = Unpacker(batch, self.config.device)

        with torch.no_grad():
            logits = self.model(unpacker.token_ids, mask=unpacker.mask)

        preds = np.argmax(
            logits.detach().cpu().numpy(), axis=2).flatten()
        labels = unpacker.get(self.attribute).cpu().numpy().flatten()
        all_preds, all_labels = [], []
        for pred, label in zip(preds, labels):
            if label not in [self.attribute.pad, self.attribute.none]:
                all_preds.append(pred)
                all_labels.append(label)

        unpacker.detach()
        return all_preds, all_labels

    def train(self):
        for epoch in range(self.epochs):
            print(f"==== Epoch {epoch + 1} ".ljust(60, "=") + "\n")

            self.model.train()
            total_loss = 0
            sleep(0.1)
            for batch in tqdm(self.config.train_loader, desc="Training"):
                total_loss += self._train(batch)
            avg_loss = total_loss / len(self.config.train_loader)
            print(f"Average loss: {avg_loss:.2}\n")

            self.model.eval()
            all_preds, all_labels = [], []
            sleep(0.1)
            for batch in tqdm(self.config.eval_loader, desc="Evaluating"):
                preds, labels = self._eval(batch)
                all_preds.extend(preds)
                all_labels.extend(labels)
            print(all_labels)
            print(all_preds)
            print(classification_report(
                all_labels, all_preds,
                labels=self.attribute.encoded_values[2:],
                target_names=self.attribute.values[2:],
                zero_division=0))
            print()
        self.model.save()


class SpatialElementTrainer:
    def __init__(self, config, epochs, lr=3e-5):
        self.config = config
        self.model = BertForTokenClassification.from_pretrained(
            "bert-base-cased",
            num_labels=iso.SpatialElement.num_values,
            output_attentions=False,
            output_hidden_states=False
        )
        self.model.to(self.config.device)
        self.epochs = epochs

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
            optimizer_grouped_parameters, lr=lr, eps=1e-8)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.config.train_loader) * self.epochs
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def _train(self, batch):
        unpacker = Unpacker(batch, self.config.device)

        self.optimizer.zero_grad()
        labels = unpacker.get(iso.SpatialElement)
        output = self.model(
            unpacker.token_ids,
            attention_mask=unpacker.mask,
            labels=labels)

        loss = output[0]

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        unpacker.detach()

        return loss.item()

    def _eval(self, batch):
        unpacker = Unpacker(batch, self.config.device)

        with torch.no_grad():
            output = self.model(
                unpacker.token_ids, attention_mask=unpacker.mask)

        preds = np.argmax(
            output[0].detach().cpu().numpy(), axis=2).flatten()
        labels = unpacker.get(iso.SpatialElement).cpu().numpy().flatten()
        all_preds, all_labels = [], []
        for pred, label in zip(preds, labels):
            if label not in [iso.SpatialElement.pad]:
                all_preds.append(pred)
                all_labels.append(label)

        unpacker.detach()
        return all_preds, all_labels

    def train(self):
        for epoch in range(self.epochs):
            print(f"==== Epoch {epoch + 1} ".ljust(60, "=") + "\n")
            self.model.train()
            total_loss = 0
            sleep(0.1)
            for batch in tqdm(self.config.train_loader, desc="Training"):
                total_loss += self._train(batch)
            avg_loss = total_loss / len(self.config.train_loader)
            print(f"Average loss: {avg_loss:.2}\n")
            self.model.eval()
            all_preds, all_labels = [], []
            sleep(0.1)
            for batch in tqdm(self.config.eval_loader, desc="Evaluating"):
                preds, labels = self._eval(batch)
                all_preds.extend(preds)
                all_labels.extend(labels)
            print(all_labels)
            print(all_preds)
            print(classification_report(
                all_labels, all_preds,
                labels=iso.SpatialElement.encoded_values[1:],
                target_names=iso.SpatialElement.values[1:],
                zero_division=0))
            print()
        self.model.save()


class Unpacker:

    def __init__(self, batch, device):
        self.device = device
        self.token_ids = batch["token_ids"].to(device)
        self.mask = batch["mask"].to(device)
        self.labels = {
            iso.SpatialElement.id: batch[iso.SpatialElement.id].to(device),
            iso.Dimensionality.id: batch[iso.Dimensionality.id].to(device),
            iso.Form.id: batch[iso.Form.id].to(device),
            iso.SemanticType.id: batch[iso.SemanticType.id].to(device),
            iso.MotionType.id: batch[iso.MotionType.id].to(device),
            iso.MotionClass.id: batch[iso.MotionClass.id].to(device),
        }

    def get(self, attribute):
        return self.labels[attribute.id]

    def detach(self):
        self.token_ids.detach()
        self.mask.detach()
        for label in self.labels.values():
            label.detach()
