from time import sleep

from tqdm import tqdm

from utils.model import Trainer


def train_model(attribute, train_loader, eval_loader, device, epochs=3):
    print("Training {} model".format(attribute.name))
    trainer = Trainer(attribute, device, train_loader, fresh=True, epochs=epochs)
    for epoch in range(epochs):
        loss = 0
        for batch in tqdm(train_loader, desc="Epoch {}".format(epoch + 1)):
            batch = tuple(t.to(device) for t in batch)
            b_ids = batch[0]
            b_labels = batch[attribute.index]
            b_mask = batch[7]
            loss += trainer.train(b_ids, b_labels, b_mask)
        total_loss = loss / len(train_loader)

        correct, total = 0, 0
        for batch in eval_loader:
            batch = tuple(t.to(device) for t in batch)
            b_ids = batch[0]
            b_labels = batch[attribute.index]
            b_mask = batch[7]
            correct, total = trainer.eval(b_ids, b_labels, b_mask)
        print("Loss: {}".format(total_loss))
        print("Accuracy: {}".format(correct / total))
        sleep(0.1)

    trainer.save()
