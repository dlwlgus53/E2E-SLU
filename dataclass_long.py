# This is longer version of dataclass
# Not use this version

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
from speech.wav2vec2_modeling import build_paired_audio, PairedAudioData

import random

# logging.set_verbosity_error()


class E2Edataclass:
    def __init__(
        self, text_tokenizer, text_data_path, audio_file_list, data_type, short=0
    ):
        self.text_tokenizer = text_tokenizer
        self.text_max_length = text_tokenizer.model_max_length
        self.text_data_path = text_data_path
        raw_text_dataset = json.load(open(text_data_path, "r"))
        self.data_type = data_type
        self.short = short

        (
            dial_id,
            turn_id,
            schema,
            question,
            target,
            belief,
        ) = self.raw_data_to_list_ratio(raw_text_dataset)

        self.dial_id = dial_id
        self.turn_id = turn_id
        self.schema = schema
        self.question = question
        self.target = target
        self.belief = belief

        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base"
        )  # Fixed
        self.audio_path_dict = self.file_list_to_dict(audio_file_list)
        # print(self.audio_path_dict)

    def __len__(self):
        return len(self.question)

    def file_list_to_dict(self, file_list_path):
        path_dict = defaultdict(lambda: defaultdict(dict))
        file = open(file_list_path, "r")
        lines = file.readlines()
        for line in lines:
            file_name = os.path.split(line.split("|")[0])[1]
            d_id = file_name.split("-")[0]
            t_id = int(file_name.split("-")[1])
            type_US = file_name.split("-")[2].replace(".wav", "")
            path_dict[d_id][t_id][type_US] = line.split("|")[0]
        return path_dict

    def raw_data_to_list(self, dataset):
        dial_id, turn_id, schema, question, target, belief = [], [], [], [], [], []
        dial_num = 0
        S = 0
        for d_id in dataset.keys():
            S += 1
            if self.short == True and S > 5:
                break
            dial_num += 1
            dial = dataset[d_id]
            for t_id, turn in enumerate(dial["log"]):
                for key in ontology.QA["all_domain"]:
                    q = ontology.QA[key]["description1"]
                    # q += turn["user"]

                    if key in turn["curr_belief"]:
                        a = turn["curr_belief"][key]
                    else:
                        a = ontology.QA["NOT_MENTIONED"]

                    question.append(q)
                    target.append(a)
                    belief.append(turn["belief"])
                    schema.append(key)
                    dial_id.append(d_id)
                    turn_id.append(t_id)

        print(f"total dial num is {dial_num}")
        return dial_id, turn_id, schema, question, target, belief

    def __getitem__(self, index):
        target = self.target[index]
        question = self.question[index]
        schema = self.schema[index]
        turn_id = self.turn_id[index]
        dial_id = self.dial_id[index]
        belief = self.dial_id[index]

        user_audio_path = self.audio_path_dict[dial_id][turn_id]["user"]

        # At first turn, there is no system utterence.
        if turn_id == 0:
            sys_audio_path = None
        else:
            sys_audio_path = self.audio_path_dict[dial_id][turn_id - 1]["system"]

        input_values_dict = build_paired_audio(
            system_path=sys_audio_path,
            user_path=user_audio_path,
            processor=self.processor,
        )

        return {
            "question": question,
            "target": target,
            "belief": belief,
            "turn_id": turn_id,
            "dial_id": dial_id,
            "schema": schema,
            "sys_audio": input_values_dict["system"],
            "user_audio": input_values_dict["user"],
        }

    def raw_data_to_list_ratio(self, dataset):
        dial_id, turn_id, schema, question, target, belief = [], [], [], [], [], []
        dial_num = 0
        S = 0
        for d_id in dataset.keys():
            S += 1
            if self.short == True and S > 5:
                break
            dial_num += 1
            dial = dataset[d_id]
            for t_id, turn in enumerate(dial["log"]):
                if len(turn["curr_belief"]) == 0:
                    key = random.choice(ontology.QA["all_domain"])
                    q = ontology.QA[key]["description1"]
                    a = ontology.QA["NOT_MENTIONED"]

                    question.append(q)
                    target.append(a)
                    belief.append(turn["belief"])
                    schema.append(key)
                    dial_id.append(d_id)
                    turn_id.append(t_id)
                else:
                    for key in ontology.QA["all_domain"]:
                        q = ontology.QA[key]["description1"]
                        # q += turn["user"]
                        if key in turn["curr_belief"]:
                            a1 = turn["curr_belief"][key]

                            a2 = ontology.QA["NOT_MENTIONED"]
                            key2 = random.choice(
                                list(
                                    set(ontology.QA["all_domain"])
                                    - set(turn["curr_belief"].keys())
                                )
                            )

                            question.append(q)
                            target.append(a1)
                            belief.append(turn["belief"])
                            schema.append(key)
                            dial_id.append(d_id)
                            turn_id.append(t_id)

                            question.append(q)
                            target.append(a2)
                            belief.append(turn["belief"])
                            schema.append(key2)
                            dial_id.append(d_id)
                            turn_id.append(t_id)

        print(f"total dial num is {dial_num}")
        return dial_id, turn_id, schema, question, target, belief

    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        """
        question = [x["question"] for x in batch]
        target = [x["target"] for x in batch]
        belief = [x["belief"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        dial_id = [x["dial_id"] for x in batch]
        schema = [x["schema"] for x in batch]

        user_audio = [x["user_audio"] for x in batch]
        sys_audio = [x["sys_audio"] for x in batch]
        gate_label = torch.tensor(
            [1 if t != ontology.QA["NOT_MENTIONED"] else 0 for t in target]
        ).to(torch.float)
        question = self.text_tokenizer.batch_encode_plus(
            question,
            max_length=self.text_max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        target = self.text_tokenizer.batch_encode_plus(
            target,
            max_length=self.text_max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        sys_input_features = [
            {"input_values": feature.input_values} for feature in sys_audio
        ]

        user_input_features = [
            {"input_values": feature.input_values} for feature in user_audio
        ]

        processed_system_audio = self.processor.pad(
            sys_input_features,
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
            return_attention_mask=True,
        )

        processed_user_audio = self.processor.pad(
            user_input_features,
            padding="longest",
            pad_to_multiple_of=None,
            return_tensors="pt",
            return_attention_mask=True,
        )

        system_audio_input = PairedAudioData(
            input_values=processed_system_audio["input_values"],
            attention_mask=processed_system_audio["attention_mask"],
        )
        user_audio_input = PairedAudioData(
            input_values=processed_user_audio["input_values"],
            attention_mask=processed_user_audio["attention_mask"],
        )
        return {
            "text_input": question,
            "label": target,
            "belief": belief,
            "gate_label": gate_label,
            "turn_id": turn_id,
            "dial_id": dial_id,
            "schema": schema,
            "user_audio_input": user_audio_input,
            "system_audio_input": system_audio_input,
        }


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
        help=" pretrainned model from ????",
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
        dataset=test_dataset, batch_size=16, collate_fn=test_dataset.collate_fn
    )

    t = test_dataset.text_tokenizer

    for batch in test_data_loader:
        print(batch.keys())

        print(batch["system_audio_input"].input_values)
        for i in range(3):
            print(
                t.decode(batch["text_input"]["input_ids"][i], skip_special_tokens=True)
            )
            print(t.decode(batch["target"]["input_ids"][i], skip_special_tokens=True))
        pdb.set_trace()
        break
