from dataset import MTDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from model import Transformer, TransformerConfig

model_config = TransformerConfig(vocab_size=32000, block_size=128, n_layer=2, n_head=4, n_embd=16, dropout=0.0, bias=True)
model = Transformer(model_config)

train_dataset = MTDataset("data/wmt/train.json")
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=train_dataset.collate_fn)

dev_dataset = MTDataset("data/wmt/dev.json")
dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=1, collate_fn=dev_dataset.collate_fn)

for epoch in range(15):
    for item in train_dataloader:
        print(item.keys())