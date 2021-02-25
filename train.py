from time import sleep

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from utils.cuda import get_device
from utils.data import AnnotatedDataset
from utils.isospace import SpatialElement, Dimensionality, Form, SemanticType, \
    MotionType, MotionClass
from utils.model import BertForSpatialElementClassification

BATCH = 8
EPOCHS = 12

device = get_device()

train_set = AnnotatedDataset("train")
eval_set = AnnotatedDataset("eval")

train_loader = DataLoader(train_set, batch_size=BATCH)
eval_loader = DataLoader(eval_set, batch_size=BATCH)

model = BertForSpatialElementClassification()
optimizer = AdamW(model.parameters(), lr=5e-3)

loss_fn = nn.CrossEntropyLoss()

model.to(device)

se_with_dimensionality = [
    SpatialElement.encode("PLACE"),
    SpatialElement.encode("PATH"),
    SpatialElement.encode("SPATIAL_ENTITY")
]

print()

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    print(f"==== Epoch {epoch + 1} ".ljust(60, "=") + "\n")

    sleep(0.1)
    for batch in tqdm(train_loader, desc="Training"):
        b_token_ids = batch["token_ids"].to(device)
        b_mask = batch["mask"].to(device)
        b_spatial_elements = batch[SpatialElement.id].to(device)
        b_dimensionality = batch[Dimensionality.id].to(device)
        b_form = batch[Form.id].to(device)
        b_semantic_type = batch[SemanticType.id].to(device)
        b_motion_type = batch[MotionType.id].to(device)
        b_motion_class = batch[MotionClass.id].to(device)

        optimizer.zero_grad()
        spatial_element_logits, dimensionality_logits = model(b_token_ids,
                                                              mask=b_mask)

        is_not_masked = b_mask.view(-1) == 1

        spatial_element_active_logits = spatial_element_logits.view(
            -1, SpatialElement.num_values)
        spatial_element_active_labels = torch.where(
            is_not_masked,
            b_spatial_elements.view(-1),
            torch.tensor(loss_fn.ignore_index).type_as(b_spatial_elements))
        spatial_element_loss = loss_fn(
            spatial_element_active_logits, spatial_element_active_labels)

        spatial_elements = b_spatial_elements.view(-1)
        is_place = torch.eq(spatial_elements, se_with_dimensionality[0])
        is_path = torch.eq(spatial_elements, se_with_dimensionality[1])
        is_spatial_entity = torch.eq(
            spatial_elements, se_with_dimensionality[2])
        has_dimensionality = torch.logical_or(
            is_place, torch.logical_or(is_path, is_spatial_entity))
        active_loss = torch.logical_and(is_not_masked, has_dimensionality)
        dimensionality_active_logits = dimensionality_logits.view(
            -1, Dimensionality.num_values)
        dimensionality_active_labels = torch.where(
            active_loss,
            b_dimensionality.view(-1),
            torch.tensor(loss_fn.ignore_index).type_as(b_dimensionality))
        dimensionality_loss = loss_fn(
            dimensionality_active_logits, dimensionality_active_labels)

        loss = spatial_element_loss + dimensionality_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        for v in [b_token_ids, b_mask, b_spatial_elements, b_dimensionality,
                  b_form, b_semantic_type, b_motion_type, b_motion_class, ]:
            v.detach()

    avg_loss = total_loss / len(train_loader)
    print(f"Average loss: {avg_loss:.2}\n")

    model.eval()

    spatial_element_all_preds, spatial_element_all_labels = [], []
    dimensionality_all_preds, dimensionality_all_labels = [], []

    sleep(0.1)
    for batch in tqdm(eval_loader, desc="Evaluating"):
        b_token_ids = batch["token_ids"].to(device)
        b_mask = batch["mask"].to(device)
        b_spatial_elements = batch[SpatialElement.id].to(device)
        b_dimensionality = batch[Dimensionality.id].to(device)
        b_form = batch[Form.id].to(device)
        b_semantic_type = batch[SemanticType.id].to(device)
        b_motion_type = batch[MotionType.id].to(device)
        b_motion_class = batch[MotionClass.id].to(device)

        with torch.no_grad():
            spatial_element_logits, dimensionality_logits = model(b_token_ids,
                                                                  mask=b_mask)

        spatial_element_preds = np.argmax(
            spatial_element_logits.detach().cpu().numpy(), axis=2).flatten()
        spatial_element_labels = b_spatial_elements.cpu().numpy().flatten()
        for pred, label in zip(spatial_element_preds, spatial_element_labels):
            if label != SpatialElement.pad:
                spatial_element_all_preds.append(pred)
                spatial_element_all_labels.append(label)

        dimensionality_preds = np.argmax(
            dimensionality_logits.detach().cpu().numpy(), axis=2).flatten()
        dimensionality_labels = b_dimensionality.cpu().numpy().flatten()
        for pred, label, se in zip(dimensionality_preds, dimensionality_labels, spatial_element_labels):
            if label != Dimensionality.pad and se in se_with_dimensionality:
                dimensionality_all_preds.append(pred)
                dimensionality_all_labels.append(label)

        for v in [b_token_ids, b_mask, b_spatial_elements, b_dimensionality,
                  b_form, b_semantic_type, b_motion_type, b_motion_class, ]:
            v.detach()

    print(classification_report(
        spatial_element_all_labels, spatial_element_all_preds,
        target_names=SpatialElement.values[1:],
        zero_division=0))
    print()

    print(classification_report(
        dimensionality_all_labels, dimensionality_all_preds,
        target_names=Dimensionality.values[1:],
        zero_division=0))
    print()

model.save()
