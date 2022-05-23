import joblib
import torch
import torch.nn.functional as F
import torchtext
from torchtext.data import get_tokenizer

from transformer_text_classifier import TweetModel

model = TweetModel(emb_dim=8).load_from_checkpoint("checkpoints/epoch=9-step=660.ckpt")
vocab: torchtext.vocab.Vocab = joblib.load("data/vocab.joblib")
tokenizer = get_tokenizer("basic_english")

model.eval()


while True:
    inp = input("Input >>> ")
    # inp = "short $tsla!!!"
    tokens = vocab(tokenizer(inp))
    mask = torch.ones(len(tokens), len(tokens))
    probas = model(torch.Tensor(tokens).long().reshape(-1, 1), mask)
    print(probas)
    # break
