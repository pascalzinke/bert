from torch.utils.data import DataLoader

from utils.cuda import get_device
from utils.data import AnnotatedDataset
from utils.train import train_model
from utils.isospace import Tag, Dimensionality, Form, SemanticType, \
    MotionType, MotionClass

BATCH = 8

device = get_device()

train_set = AnnotatedDataset("train")
eval_set = AnnotatedDataset("eval")

train_loader = DataLoader(train_set, batch_size=BATCH)
eval_loader = DataLoader(eval_set, batch_size=BATCH)

for attribute in [Tag, Dimensionality, Form, SemanticType, MotionType,
                  MotionClass]:
    train_model(attribute, train_loader, eval_loader, device)
