from torch.utils.data import DataLoader

import utils.isospace as iso
from utils.cuda import get_device
from utils.data import TextDataset
from utils.train import Trainer, TrainerConfig

BATCH = 8

device = get_device()

train_set = TextDataset()
eval_set = TextDataset(evaluate=True)

train_loader = DataLoader(train_set, batch_size=BATCH)
eval_loader = DataLoader(eval_set, batch_size=BATCH)

config = TrainerConfig(device, train_loader, eval_loader)

spatial_element_trainer = Trainer(iso.SpatialElement, config, keep_none=True)
dimensionality_trainer = Trainer(iso.Dimensionality, config)
form_trainer = Trainer(iso.Form, config)
semantic_type_trainer = Trainer(iso.SemanticType, config)
motion_type_trainer = Trainer(iso.MotionType, config)
motion_class_trainer = Trainer(iso.MotionClass, config)

spatial_element_trainer.train()
dimensionality_trainer.train()
form_trainer.train()
semantic_type_trainer.train()
motion_type_trainer.train()
motion_class_trainer.train()
