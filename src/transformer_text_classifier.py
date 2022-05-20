import joblib
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchtext
from positional_encodings import PositionalEncoding1D, Summer
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch import nn, utils
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

logger = TensorBoardLogger("logs", name="my_model")
tokenizer = get_tokenizer("basic_english")


class TweetDataset(utils.data.Dataset):
    def __init__(self) -> None:
        self.x = []
        self.y = []

        df = pd.read_csv("data/SRS_sentiment_labeled.csv")
        self.x = df["tweet"].to_numpy()
        self.y = df["sentiment"].to_numpy() + 1
        self.xtrain, self.xval, self.ytrain, self.yval = train_test_split(
            self.x, self.y, random_state=42
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)


def collate_batch(batch):
    x_processed = []
    y_processed = []
    for x, y in batch:
        x_processed.append(torch.Tensor(vocab(tokenizer(x))).long())
        y_processed.append(y)
    return (
        nn.utils.rnn.pad_sequence(x_processed),
        torch.Tensor(y_processed).long().flatten(),
        # TODO: create attention mask re: padding
    )


# pytorch lightning module
class TweetModel(pl.LightningModule):
    def __init__(self, emb_dim: int = 16, lr: float = 2e-3) -> None:
        super().__init__()
        self.lr = lr
        self.accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.emb_dim = emb_dim
        self.add_pos_encoding = Summer(PositionalEncoding1D(emb_dim))
        self.embedding = nn.Embedding(len(vocab), emb_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=1, dim_feedforward=128
        )

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=self.emb_dim, out_features=3)

    def forward(self, x):
        x = self.embedding(x)
        x = self.add_pos_encoding(x)
        x = self.transformer(x)
        x = F.dropout(x, 0.5)
        x = torch.mean(x, 0)
        # x = F.dropout(x, 0.5)
        x = self.dense(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss)
        self.accuracy(y_hat, y)
        self.log("train_acc", self.accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_accuracy(y_hat, y)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

        return loss

    # def training_epoch_end(self, outputs):
    #     self.log("train_acc", self.accuracy)

    # def val_epoch_end(self, outputs):
    #     self.log("val_acc", self.val_accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class TweetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, rebuild_vocab: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.rebuild_vocab = rebuild_vocab

    def setup(self, stage):
        self.dataset = TweetDataset()
        self._train_size = int(len(self.dataset) * 0.7)
        self._val_size = len(self.dataset) - self._train_size
        print(f"train size, val size = ({self._train_size}, {self._val_size})")

        self.train_dataset, self.val_dataset = utils.data.random_split(
            self.dataset, [self._train_size, self._val_size]
        )

        if self.rebuild_vocab:
            vocab = build_vocab_from_iterator(
                yield_tokens(self.train_dataset), specials=["<unk>"], min_freq=5
            )
            vocab.set_default_index(vocab["<unk>"])
            joblib.dump(vocab, "data/vocab.joblib")
            print("Saved vocab.")

    def train_dataloader(self):
        return utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=12,
        )

    def val_dataloader(self):
        return utils.data.DataLoader(
            self.val_dataset,
            batch_size=1024,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=12,
        )


vocab: torchtext.vocab.Vocab = joblib.load("data/vocab.joblib")
print("Loaded vocab.")
print(f"vocab size = {len(vocab)}")


model = TweetModel(emb_dim=8)
trainer = pl.Trainer(
    logger=logger, max_epochs=50, log_every_n_steps=40, auto_lr_find=False
)

trainer.fit(model, TweetDataModule(batch_size=32))
