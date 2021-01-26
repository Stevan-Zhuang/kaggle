import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl

train = pd.read_csv(r"..\input\digit-recognizer\train.csv")
x_test = pd.read_csv(r"..\input\digit-recognizer\test.csv")

y_train = train['label']
x_train = train.drop('label', axis=1)

test_id = x_test.index

x_train = torch.tensor(x_train.values, dtype=torch.float)
y_train = torch.tensor(y_train.values, dtype=torch.long)
x_test = torch.tensor(x_test.values, dtype=torch.float)

dataset = TensorDataset(x_train, y_train)

class LitModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.nll_loss(y_pred, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2)

model = LitModel()
trainer = pl.Trainer(max_epochs=1)

trainer.fit(model, DataLoader(dataset, batch_size=64))

num_correct = 0
for x, y in dataset:
    y_pred = model(x.view(1, 784))
    if y_pred.argmax(1).item() == y:
        num_correct += 1
print(num_correct / 42000)

y_pred = model(x_test).argmax(1)

submission = pd.DataFrame({'ImageId': test_id + 1, 'Label': y_pred.numpy()})
submission.to_csv(r"submission.csv", index=False)
