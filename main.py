import os
import pdb
import sys
import json
import torch
import random
import argparse
from collections import OrderedDict
from logger_conf import CreateLogger
from transformers import AutoModel, AutoConfig, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
from dataclass import E2Edataclass
from logger_conf import CreateLogger

from transformers import T5Tokenizer, Adafactor
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from trainer import Trainer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from evaluate import acc_metric

from model import Our_Transformer

parser = argparse.ArgumentParser()

# data setting


parser.add_argument("--text_train_path", type=str, default="./woz_data/train_data.json")
parser.add_argument("--text_dev_path", type=str, default="./woz_data/dev_data.json")
parser.add_argument("--text_test_path", type=str, default="./woz_data/test_data.json")
parser.add_argument("--audio_train_list", type=str)
parser.add_argument("--audio_dev_list", type=str)
parser.add_argument("--audio_test_list", type=str)

# training setting
parser.add_argument("--short", type=int, default=0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_epoch", type=int, default=1)
parser.add_argument("--gpus", default=1, type=int, help="number of gpus per node")
parser.add_argument("--save_prefix", type=str, default="")
parser.add_argument("--patient", type=int, help="prefix for all savings", default=3)

# model parameter
parser.add_argument("--batch_size_per_gpu", type=int, default=16)
parser.add_argument("--test_batch_size_per_gpu", type=int, default=16)
parser.add_argument("--use_fine_trained", type=str)
parser.add_argument("--model_config", type=str, default="./configs/transformer.json")
parser.add_argument("--text_encoder_freeze", type=int, default=0)
parser.add_argument("--audio_encoder_freeze", type=int, default=0)
parser.add_argument("--text_encoder_model", type=str, default="bert-base-uncased")


def init_experiment(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU


def load_trained(model, model_path):
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace(
            "module.", ""
        )  # [7:]remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(f"./logs/{args.save_prefix}", exist_ok=True)
    os.makedirs("./out", exist_ok=True)
    os.makedirs(f"model/{args.save_prefix}", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🍪 DEVICE: {device}")

    init_experiment(args.seed)
    log_folder = f"logs/{args.save_prefix}/"
    logger = CreateLogger("main", os.path.join(log_folder, "info.log"))
    logger.info("------------------------START NEW TRAINING-----------------")
    if args.short == 1:
        args.max_epoch = 1

    logger.info(args)
    writer = SummaryWriter()

    model = Our_Transformer(
        d_model=768,
        text_encoder=args.text_encoder_model,
        transformer_config=AutoConfig.from_pretrained(args.model_config),
        device=device,
    ).to(device)

    # Freezing #  TODO as option?
    if args.text_encoder_freeze:
        model.TextEncoder.requires_grad_(False)
    if args.audio_encoder_freeze:
        model.AudioEncoder.requires_grad_(False)

    if args.use_fine_trained:
        model = load_trained(model, args.fine_trained)

    if args.gpus > 1:
        model = nn.DataParallel(model)

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_model)
    train_dataset = E2Edataclass(
        text_tokenizer,
        args.text_train_path,
        args.audio_train_list,
        "train",
        short=args.short,
    )
    dev_dataset = E2Edataclass(
        text_tokenizer,
        args.text_dev_path,
        args.audio_dev_list,
        "dev",
        short=args.short,
    )

    test_dataset = E2Edataclass(
        text_tokenizer,
        args.text_test_path,
        args.audio_test_list,
        "test",
        short=args.short,
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=4, collate_fn=test_dataset.collate_fn
    )

    optimizer_setting = {
        "warmup_init": True,
        "eps": (1e-30, 1e-3),
        "clip_threshold": 1.0,
        "decay_rate": -0.8,
        "beta1": None,
        "weight_decay": 0.0,
        "relative_step": True,
        "scale_parameter": False,
    }

    optimizer = Adafactor(model.parameters(), **optimizer_setting)

    trainer_setting = {
        "train_batch_size": args.batch_size_per_gpu * args.gpus,
        "test_batch_size": args.test_batch_size_per_gpu * args.gpus,
        "tokenizer": text_tokenizer,
        "log_folder": log_folder,
        "save_prefix": args.save_prefix,
        "max_epoch": args.max_epoch,
    }

    model_trainer = Trainer(
        model=model,
        train_data=train_dataset,
        valid_data=dev_dataset,
        test_data=test_dataset,
        optimizer=optimizer,
        logger_name="model",
        evaluate_fnc=acc_metric,
        **trainer_setting,
    )

    model_trainer.work(test=True, save=True, train=True)
