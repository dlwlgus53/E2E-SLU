import torch
from torch import nn, Tensor
from dataclass import E2Edataclass
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel
from torch import nn
from transformers import Trainer
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model, AutoTokenizer
from transformers import logging
import config as cfg
import argparse
import ontology
import os
from transformers import AutoModel, AutoConfig, Wav2Vec2Processor
from speech.wav2vec2_modeling import build_paired_audio
from speech.wav2vec2_modeling import AudioEncoder
from transformers import TrainingArguments, Trainer
from logger_conf import CreateLogger


from transformers import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
    PreTrainedModel,
    PretrainedConfig,
)


class TextEncoder(nn.Module):
    def __init__(self, model_name, d_model, device):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        emb_size = self.bert.config.hidden_size
        self.fc1 = nn.Linear(emb_size, d_model)
        self.device = device

    def forward(self, text_data):
        input_ids, attention_mask = (
            text_data["input_ids"].to(self.device),
            text_data["attention_mask"].to(self.device),
        )
        embeded = self.get_embeddings(input_ids, attention_mask)
        return embeded

    def get_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(
            input_ids=input_ids, attention_mask=attention_mask
        )  # [0] : B x L x H(768)

        # attention_mask = attention_mask.unsqueeze(-1) # B x L x 1

        # mean_output = torch.sum(bert_output[0] * attention_mask, dim=1) / torch.sum(
        #     attention_mask, dim=1
        # ) # B x H(768)
        # out = self.fc1(bert_output[0])  # B x H(512)

        return {
            "embedding": bert_output.pooler_output,
            "attention_mask": attention_mask,
        }


class Our_Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        text_encoder: str,
        transformer_config,
        device,  # emb size, 512
    ):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.TextEncoder = TextEncoder(text_encoder, d_model, device).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        audio_config = PretrainedConfig()
        self.AudioEncoder = AudioEncoder(audio_config, output_dim=d_model).to(device)
        # self.AudioEncoder.config

        EDconfig = EncoderDecoderConfig.from_encoder_decoder_configs(
            transformer_config, transformer_config  # config can be changed
        )
        self.transformer = EncoderDecoderModel(config=EDconfig)
        self.transformer.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.transformer.config.pad_token_id = self.tokenizer.pad_token_id

    def audio_to_device(self, audio):
        audio.input_values = audio.input_values.to(self.device)
        audio.attention_mask = audio.attention_mask.to(self.device)

    def generate(self, batch):
        output = self.transformer.generate(
            input_ids=batch["text_input"]["input_ids"].to(self.device),  # add audio
            attention_mask=batch["text_input"]["attention_mask"].to(self.device),
        )
        return output

    def forward(self, batch) -> Tensor:

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
                text_src["embedding"].unsqueeze(1),
                sys_audio_src["embeddings"].unsqueeze(1),
                user_audio_src["embeddings"].unsqueeze(1),
            ],
            axis=1,
        )  # B x L x H
        import pdb

        # text_attention_mask = text_src["attention_mask"].unsqueeze(1)
        # mask = torch.cat(
        #     [
        #         text_attention_mask,
        #         torch.ones([len(text_attention_mask), 2]).to(self.device),
        #     ],
        #     axis=1,
        # )

        batch["label"]["input_ids"] = [
            [
                -100 if token == self.tokenizer.pad_token_id else token
                for token in labels
            ]
            for labels in batch["label"]["input_ids"]
        ]

        batch["label"]["input_ids"] = torch.tensor(batch["label"]["input_ids"])
        output = self.transformer(
            inputs_embeds=src,  # add audio
            labels=batch["label"]["input_ids"].to(self.device),
        )
        # attention_mask=mask,  # add attention

        # output = self.transformer(
        #     input_ids=batch["text_input"]["input_ids"].to(self.device),  # add audio
        #     labels=batch["label"]["input_ids"].to(self.device),
        # )

        return output


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
    # This is for debugging
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

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🍪 DEVICE: {device}")

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

    model = Our_Transformer(
        d_model=768,
        text_encoder="bert-base-uncased",
        transformer_config=AutoConfig.from_pretrained("./configs/transformer.json"),
    ).to(device)
