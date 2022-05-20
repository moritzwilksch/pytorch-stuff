import joblib
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchtext
from positional_encodings import PositionalEncoding1D, Summer
from pytorch_lightning.loggers import TensorBoardLogger
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
    )


# pytorch lightning module
class TweetModel(pl.LightningModule):
    def __init__(self, context_size: int, emb_dim: int = 16, lr: float = 1e-2) -> None:
        super().__init__()
        self.context_size = context_size
        self.lr = lr
        self.accuracy = torchmetrics.Accuracy()
        self.emb_dim = emb_dim
        self.add_pos_encoding = Summer(PositionalEncoding1D(emb_dim))
        self.embedding = nn.Embedding(len(vocab), emb_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=2, dim_feedforward=128
        )

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=self.emb_dim, out_features=len(vocab))

    def forward(self, x):
        x = self.embedding(x)
        x = self.add_pos_encoding(x)
        x = self.transformer(x)
        x = torch.mean(x, 0)
        x = self.dense(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss)
        self.accuracy(y_hat, y)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        self.log("train_acc", self.accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


CONTEXT_SIZE = 32
ds = TweetDataset()

REBUILD = True
if REBUILD:
    vocab = build_vocab_from_iterator(yield_tokens(ds), specials=["<unk>"], min_freq=5)
    vocab.set_default_index(vocab["<unk>"])
    joblib.dump(vocab, "data/vocab.joblib")
    print("Saved vocab.")
else:
    vocab: torchtext.vocab.Vocab = joblib.load("data/vocab.joblib")
    print("Loaded vocab.")

print(f"vocab size = {len(vocab)}")


train_loader = utils.data.DataLoader(
    ds, batch_size=32, shuffle=True, collate_fn=collate_batch, num_workers=12
)

model = TweetModel(context_size=CONTEXT_SIZE)
trainer = pl.Trainer(
    logger=logger, max_epochs=50, log_every_n_steps=40, auto_lr_find=False
)
# trainer.tune(model, train_loader)
trainer.fit(model, train_loader)


print("end")
