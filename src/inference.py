from lib2to3.pgen2 import token
from transformer_text_classifier import TweetModel
import joblib
import torchtext
import torch
import torch.nn.functional as F
from torchtext.data import get_tokenizer

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
