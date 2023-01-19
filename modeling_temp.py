import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from dataclass import E2Edataclass

import copy
import time
import re
import pdb
import json
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model, AutoTokenizer
from transformers import logging
import config as cfg
import argparse
from collections import defaultdict
import ontology
import os
from transformers import AutoModel, AutoTokenizer, AutoConfig, Wav2Vec2Processor
from speech.wav2vec2_modeling import build_paired_audio
from speech.wav2vec2_modeling import AudioEncoder

from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
    PreTrainedModel,
    PretrainedConfig,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name,
    ):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.emb_size = self.bert.config.hidden_size

    def forward(self, text_data):
        input_ids, attention_mask = (
            text_data["input_ids"].cuda(),
            text_data["attention_mask"].cuda(),
        )
        embeded = self.get_mean_embeddings(input_ids, attention_mask)
        return embeded

    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(
            input_ids=input_ids, attention_mask=attention_mask
        )
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0] * attention_mask, dim=1) / torch.sum(
            attention_mask, dim=1
        )
        return mean_output


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        text_encoder: str,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.TextEncoder = TextEncoder(text_encoder).cuda()
        config = PretrainedConfig()
        self.AudioEncoder = AudioEncoder(config).cuda()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, batch) -> Tensor:

        text_src = self.TextEncoder(batch["text_input"])  # B x H(768)
        audio_data = {
            "user": batch["user_audio_input"],
            "system": batch["system_audio_input"],
        }
        audio_src = self.AudioEncoder(audio_data)
        pdb.set_trace()

        # src = text_src + audio_src
        # output = self.transformer_encoder(src, src_mask)
        # output = self.decoder(output)
        output = 1
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


# def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
#     """Converts raw text into a flat Tensor."""
#     data = [
#         torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter
#     ]
#     return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


# train_iter was "consumed" by the process of building the vocab,
# so we have to create it again
# train_iter, val_iter, test_iter = WikiText2()
# train_data = data_process(train_iter)
# val_data = data_process(val_iter)
# test_data = data_process(test_iter)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[: seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


batch_size = 20
eval_batch_size = 10
# train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
# val_data = batchify(val_data, eval_batch_size)
# test_data = batchify(test_data, eval_batch_size)

bptt = 1


def train(model: nn.Module) -> None:
    model.train()  # 학습 모드 시작
    total_loss = 0.0
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    for i, batch in enumerate(test_data_loader):

        output = model(batch)
        loss = criterion(output.view(-1, ntokens), batch["target"])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(
                f"| epoch {epoch:3d} | {batch:5d}/{10:5d} batches | "
                f"lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | ppl {ppl:8.2f}"
            )
            total_loss = 0
            start_time = time.time()


# def evaluate(model: nn.Module, eval_data: Tensor) -> float:
#     model.eval()  # 평가 모드 시작
#     total_loss = 0.0
#     src_mask = generate_square_subsequent_mask(bptt).to(device)
#     with torch.no_grad():
#         for i in range(0, eval_data.size(0) - 1, bptt):
#             data, targets = get_batch(eval_data, i)
#             batch_size = data.size(0)
#             if batch_size != bptt:
#                 src_mask = src_mask[:batch_size, :batch_size]
#             output = model(data, src_mask)
#             output_flat = output.view(-1, ntokens)
#             total_loss += batch_size * criterion(output_flat, targets).item()
#     return total_loss / (len(eval_data) - 1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_test_data_path", type=str, default="./woz_data/dev_data.json"
    )
    parser.add_argument(
        "--audio_test_file_list", type=str, default="finalfilelist-dev_all.txt"
    )
    parser.add_argument(
        "--text_encoder_model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help=" pretrainned model from 🤗",
    )
    parser.add_argument(
        "--data_path_prefix",
        type=str,
        default="Avoid absolute path",
    )

    args = parser.parse_args()
    text_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_model)

    test_dataset = E2Edataclass(
        text_tokenizer,
        args.text_test_data_path,
        args.audio_test_file_list,
        "test",
        short=0,
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn
    )

    best_val_loss = float("inf")
    epochs = 3
    best_model = None

    ntokens = text_tokenizer.vocab_size  # 단어 사전(어휘집)의 크기
    emsize = 200  # 임베딩 차원
    d_hid = 200  # nn.TransformerEncoder 에서 피드포워드 네트워크(feedforward network) 모델의 차원
    nlayers = 2  # nn.TransformerEncoder 내부의 nn.TransformerEncoderLayer 개수
    nhead = 2  # nn.MultiheadAttention의 헤드 개수
    dropout = 0.2  # 드랍아웃(dropout) 확률
    model = TransformerModel(
        ntokens, emsize, nhead, d_hid, nlayers, args.text_encoder_model, dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # 학습률(learning rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model)
        # val_loss = evaluate(model, val_data)
        # val_ppl = math.exp(val_loss)
        # elapsed = time.time() - epoch_start_time
        # print("-" * 89)
        # print(
        #     f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        #     f"valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}"
        # )
        # print("-" * 89)

        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     best_model = copy.deepcopy(model)

        # scheduler.step()
