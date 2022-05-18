#%%
from typing import Any

import pandas as pd
import pytorch_lightning as ptl
import seaborn as sns
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset

df = pd.get_dummies(sns.load_dataset("tips"))
df["total_bill"] = (df["total_bill"] - df["total_bill"].mean()) / df["total_bill"].std()
print(df.shape)
input_dim = df.shape[1] - 1

logger = ptl.loggers.tensorboard.TensorBoardLogger("tb_logs", name="my_model")


class MyDataset(Dataset):
    def __init__(self):
        df = pd.get_dummies(sns.load_dataset("tips"))
        df["total_bill"] = (df["total_bill"] - df["total_bill"].mean()) / df[
            "total_bill"
        ].std()

        self.x = df.drop("tip", axis=1).to_numpy()
        self.y = df["tip"].to_numpy().ravel()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MyModel(ptl.LightningModule):
    def __init__(self, lr: float = 0.001, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.learning_rate = lr
        self.activation = nn.ReLU()

        self.ff = nn.Sequential(
            nn.Linear(input_dim, 128),
            self.activation,
            nn.Dropout(),
            nn.Linear(128, 128),
            self.activation,
            nn.Dropout(),
            nn.Linear(128, 1),
        )

    def training_step(self, batch):
        x, y = batch
        x, y = x.float(), y.float()
        yhat = self.ff(x)
        loss_fn = nn.MSELoss()
        loss = loss_fn(yhat.flatten(), y.flatten())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.ff(x.float())
        loss = nn.MSELoss()(yhat.flatten(), y.float().flatten())
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


dataset = MyDataset()
train_set, val_set = torch.utils.data.random_split(dataset, [200, 44])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, shuffle=False, batch_size=100)

model = MyModel()
trainer = ptl.Trainer(max_epochs=250, auto_lr_find=True, logger=logger)
# lrfinder = trainer.tune(model, train_loader)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
