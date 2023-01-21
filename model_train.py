import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.data import dataset
from dataclass import E2Edataclass
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel
from torch import nn
from transformers import Trainer
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
from transformers import AutoModel, AutoConfig, Wav2Vec2Processor
from speech.wav2vec2_modeling import build_paired_audio
from speech.wav2vec2_modeling import AudioEncoder
from transformers import TrainingArguments, Trainer


from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
    PreTrainedModel,
    PretrainedConfig,
)


class TextEncoder(nn.Module):
    def __init__(self, model_name, d_model):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        emb_size = self.bert.config.hidden_size
        self.fc1 = nn.Linear(emb_size, d_model)

    def forward(self, text_data):
        input_ids, attention_mask = (
            text_data["input_ids"].to(device),
            text_data["attention_mask"].to(device),
        )
        embeded = self.get_mean_embeddings(input_ids, attention_mask)
        return embeded

    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(
            input_ids=input_ids, attention_mask=attention_mask
        )  # [0] : B x L x H(768)

        # attention_mask = attention_mask.unsqueeze(-1) # B x L x 1

        # mean_output = torch.sum(bert_output[0] * attention_mask, dim=1) / torch.sum(
        #     attention_mask, dim=1
        # ) # B x H(768)
        out = self.fc1(bert_output[0])  # B x H(512)
        return {"embedding": out, "attention_mask": attention_mask}


class Our_Transformer(nn.Module):
    def __init__(
        self, d_model: int, text_encoder: str, transformer_config  # emb size, 512
    ):
        super().__init__()
        self.d_model = d_model
        self.TextEncoder = TextEncoder(text_encoder, d_model).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        audio_config = PretrainedConfig()
        self.AudioEncoder = AudioEncoder(audio_config, output_dim=d_model).to(device)
        self.AudioEncoder.config

        EDconfig = EncoderDecoderConfig.from_encoder_decoder_configs(
            transformer_config, transformer_config  # config can be changed
        )
        self.transformer = EncoderDecoderModel(config=EDconfig)
        self.transformer.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.transformer.config.pad_token_id = self.tokenizer.pad_token_id

    def audio_to_device(self, audio):
        audio.input_values = audio.input_values.to(device)
        audio.attention_mask = audio.attention_mask.to(device)

    def forward(self, **batch) -> Tensor:

        text_src = self.TextEncoder(batch["text_input"])  # B x H(512)
        self.audio_to_device(batch["user_audio_input"])
        self.audio_to_device(batch["system_audio_input"])

        audio_data = {
            "user": batch["user_audio_input"],
            "system": batch["system_audio_input"],
        }
        sys_audio_src, user_audio_src = self.AudioEncoder(audio_data)
        src = torch.cat(
            [
                text_src["embedding"],
                sys_audio_src["embeddings"].unsqueeze(1),
                user_audio_src["embeddings"].unsqueeze(1),
            ],
            axis=1,
        )  # B x L x H

        mask = torch.cat(
            [
                text_src["attention_mask"],
                torch.ones([len(text_src["attention_mask"]), 2]).to(device),
            ],
            axis=1,
        )
        output = self.transformer(
            inputs_embeds=src,  # add audio
            attention_mask=mask,  # add attention
            labels=batch["target"]["input_ids"],
        )

        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🍪 DEVICE: {device}")
# device = "cpu"


import numpy as np


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


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
        default="bert-base-uncased",
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
        dataset=test_dataset, batch_size=4, collate_fn=test_dataset.collate_fn
    )

    best_val_loss = float("inf")
    epochs = 3
    best_model = None

    # ntokens = text_tokenizer.vocab_size  # 단어 사전(어휘집)의 크기
    # emsize = 512  # 임베딩 차원
    # d_hid = 512  # nn.TransformerEncoder 에서 피드포워드 네트워크(feedforward network) 모델의 차원
    # nlayers = 2  # nn.TransformerEncoder 내부의 nn.TransformerEncoderLayer 개수
    # nhead = 2  # nn.MultiheadAttention의 헤드 개수
    # dropout = 0.2  # 드랍아웃(dropout) 확률
    model = Our_Transformer(
        d_model=768,
        text_encoder="bert-base-uncased",
        transformer_config=AutoConfig.from_pretrained("./configs/transformer.json"),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # 학습률(learning rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch"
    )
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

training_args = Seq2SeqTrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    logging_steps=100,
    save_steps=500,
    save_total_limit=1,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=test_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=test_dataset.collate_fn,
)

trainer.train()
