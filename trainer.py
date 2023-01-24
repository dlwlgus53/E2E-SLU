import os
import torch
import pdb
import json
import logging
import ontology
from logger_conf import CreateLogger
from collections import defaultdict
from collections import OrderedDict
import copy
from transformers import AutoModel

# from utils import save_pickle
class Trainer:
    def __init__(
        self,
        model,
        train_batch_size,
        test_batch_size,
        tokenizer,
        optimizer,
        scheduler,
        log_folder,
        save_prefix,
        max_epoch,
        logger_name,
        evaluate_fnc,
        train_data=None,
        valid_data=None,
        test_data=None,
        patient=3,
    ):
        self.log_folder = log_folder
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = CreateLogger(logger_name, os.path.join(log_folder, "info.log"))
        self.save_prefix = save_prefix
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.model = model
        self.max_epoch = max_epoch
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.patient = patient
        self.evaluate_fnc = evaluate_fnc
        os.makedirs(f"model/{self.save_prefix}", exist_ok=True)

    def work(
        self, train_data=None, test=False, train=True, save=False, model_path=None
    ):

        if train == False and model_path == None:
            self.logger.error("train is False and model_path is NOne")
        if train_data:
            self.train_data = train_data
        min_loss = float("inf")
        best_model = ""

        if train:
            try_ = 0
            for epoch in range(self.max_epoch):
                try_ += 1
                self.train(epoch)
                loss = self.valid(epoch)
                if loss < min_loss:
                    try_ = 0
                    min_loss = loss
                    best_model = copy.deepcopy(self.model)  # deep copy

                if try_ > self.patient:
                    self.logger.info(f"Early stop in Epoch {epoch}")
                    break

            self.model = copy.deepcopy(best_model)
            if save == True:
                torch.save(
                    self.model.state_dict(),
                    f"model/{self.save_prefix}/epoch_{epoch}_loss_{min_loss:.4f}.pt",
                )

        if test == True:
            # if self.belief_type:
            #     gold, pred = self.belief_test()
            # else:
            gold, pred, pred_dict = self.test()
            self.save_test_result(
                pred_dict, self.test_data.data_path, f"out/{self.save_prefix}"
            )
            return 1

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def init_model(self, base_model):
        self.model = AutoModel.from_pretrained(base_model, return_dict=True)

    def train(self, epoch_num):
        train_max_iter = int(len(self.train_data) / self.train_batch_size)
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_data,
            batch_size=self.train_batch_size,
            collate_fn=self.train_data.collate_fn,
        )

        loss_sum = 0
        self.model.train()

        for iter, batch in enumerate(train_loader):
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = outputs.loss.mean()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss_sum += loss.detach().item()
            if (iter + 1) % 10 == 0:
                self.logger.info(self.optimizer.param_groups[0]["lr"])

                self.logger.info(
                    f"Epoch {epoch_num} training : {iter+1}/{train_max_iter } loss : {loss_sum/50:.4f}"
                )
                outputs_idx = torch.argmax(outputs.logits, 2)
                loss_sum = 0

                if (iter + 1) % 50 == 0:
                    predict_text = self.tokenizer.batch_decode(
                        outputs_idx, skip_special_tokens=True
                    )

                    batch["label"]["input_ids"][
                        batch["label"]["input_ids"] == -100
                    ] = self.tokenizer.pad_token_id
                    answer_text = self.tokenizer.batch_decode(
                        batch["label"]["input_ids"], skip_special_tokens=True
                    )
                    self.logger.info(f"ans  : {answer_text[:]}")
                    self.logger.info(f"pred : {predict_text[:]}")

                    # question_text = tokenizer.batch_decode(batch['input']['input_ids'], skip_special_tokens = True)
                    # '\n'.join(question_text),iter)

    def valid(self, epoch_num):
        valid_max_iter = int(len(self.valid_data) / self.test_batch_size)
        valid_loader = torch.utils.data.DataLoader(
            dataset=self.valid_data,
            batch_size=self.test_batch_size,
            collate_fn=self.valid_data.collate_fn,
        )
        self.model.eval()
        loss_sum = 0
        self.logger.info("Validation Start")
        with torch.no_grad():
            for iter, batch in enumerate(valid_loader):
                outputs = self.model(batch)
                loss = outputs.loss.mean()

                loss_sum += loss.detach()
                if (iter + 1) % 50 == 0:
                    self.logger.info(
                        f"Epoch {epoch_num} Validation : {iter+1}/{valid_max_iter}"
                    )

        self.logger.info(f"Epoch {epoch_num} Validation loss : {loss_sum/iter:.4f}")
        return loss_sum / iter

    def test(self):
        answer, pred = [], []
        pred_dict = defaultdict(dict)
        test_max_iter = int(len(self.test_data) / self.test_batch_size)
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_data,
            batch_size=self.test_batch_size,
            collate_fn=self.test_data.collate_fn,
        )
        result_dict = defaultdict(
            lambda: defaultdict(dict)
        )  # dial_id, # turn_id # schema
        self.model.eval()
        loss_sum = 0
        self.logger.info("Test start")
        with torch.no_grad():
            for iter, batch in enumerate(test_loader):
                if iter % 10 == 0:
                    self.logger.info(f"Test... {iter+1}/{test_max_iter}")
                outputs = self.model(batch)
                outputs_idx = torch.argmax(outputs.logits, 2)

                predict_text = self.tokenizer.batch_decode(
                    outputs_idx, skip_special_tokens=True
                )
                batch["label"]["input_ids"][
                    batch["label"]["input_ids"] == -100
                ] = self.tokenizer.pad_token_id
                answer_text = self.tokenizer.batch_decode(
                    batch["label"]["input_ids"], skip_special_tokens=True
                )

                pred.extend(predict_text)
                answer.extend(answer_text)
                for d, t, p in zip(batch["dial_id"], batch["turn_id"], predict_text):
                    pred_dict[d][t] = p

        pred_dict = dict(pred_dict)
        return answer, pred, pred_dict

    def string_to_dict(slef, belief_str):
        belief_dict = {}
        items = belief_str.split(",")
        for item in items:
            try:
                key, value = item.split(": ")
                belief_dict[key.strip()] = value.strip()
            except ValueError:
                continue
        return belief_dict

    def save_test_result(self, result_dict, test_data_path, save_path):

        test_dataset = json.load(open(test_data_path, "r"))

        for d_id in test_dataset.keys():
            dial = test_dataset[d_id]["log"]
            for turn in dial:
                t_id = int(turn["turn_num"])
                try:
                    turn.update({"pred": result_dict[d_id][t_id]})
                except:
                    pass

        with open(save_path, "w") as f:
            json.dump(test_dataset, f, ensure_ascii=False, indent=4)

    def belief_test(self):
        test_max_iter = int(len(self.test_data) / self.test_batch_size)
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_data,
            batch_size=self.test_batch_size,
            collate_fn=self.test_data.collate_fn,
        )

        belief_state = defaultdict(
            lambda: defaultdict(dict)
        )  # dial_id, # turn_id # schema
        self.model.eval()
        loss_sum = 0
        self.logger.info("Test start")
        with torch.no_grad():
            for iter, batch in enumerate(test_loader):
                outputs_text = self.model.module.generate(
                    input_ids=batch["input"]["input_ids"].to("cuda")
                )
                outputs_text = self.tokenizer.batch_decode(
                    outputs_text, skip_special_tokens=True
                )
                for idx in range(len(outputs_text)):
                    dial_id = batch["dial_id"][idx]
                    turn_id = batch["turn_id"][idx]
                    belief_state[dial_id][turn_id] = self.string_to_dict(
                        outputs_text[idx]
                    )
                if (iter + 1) % 50 == 0:
                    self.logger.info(f"Test : {iter}/{test_max_iter}")
        answer = self.test_data.get_data()
        return answer, belief_state
