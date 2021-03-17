from torch.utils.data import DataLoader

import utils.isospace as iso
from utils.cuda import get_device
from utils.data import AnnotatedDataset
from utils.train import AttributeTrainer, TrainerConfig, SpatialElementTrainer

BATCH = 8
EPOCHS = 20

device = get_device()

train_set = AnnotatedDataset("train")
eval_set = AnnotatedDataset("eval")

train_loader = DataLoader(train_set, batch_size=BATCH)
eval_loader = DataLoader(eval_set, batch_size=BATCH)

config = TrainerConfig(device, train_loader, eval_loader)

# spatial_element_trainer = SpatialElementTrainer(config, 2)
# dimensionality_trainer = Trainer(iso.Dimensionality, config, 12)
form_trainer = AttributeTrainer(iso.Form, config, 2)
# semantic_type_trainer = Trainer(iso.SemanticType, config, 12)
# motion_type_trainer = Trainer(iso.MotionType, config, 12)
# motion_class_trainer = Trainer(iso.MotionClass, config, 12)

form_trainer.train()
