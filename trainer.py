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
from torch import nn
from evaluate import evaluate_metrics

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
        device="cuda",
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
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.gate_loss_fn = nn.BCELoss()
        self.device = device

        os.makedirs(f"model/{self.save_prefix}", exist_ok=True)

    def work(self, train_data=None, test=False, train=True, save=False, alpha=0.5):
        # TODO need clean here

        if train_data:
            self.train_data = train_data
        min_loss = float("inf")
        best_model = ""

        if train:
            try_ = 0
            for epoch in range(self.max_epoch):
                try_ += 1
                self.train(epoch, alpha)
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
            answer, pred, pred_gate_list, answer_gate_list, pred_dict = self.test()
            self.save_test_result(
                pred_dict, self.test_data.text_data_path, f"out/{self.save_prefix}"
            )
            evaluate_result = evaluate_metrics(pred_dict, self.test_data.text_data_path)
            return evaluate_result

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def init_model(self, base_model):
        self.model = AutoModel.from_pretrained(base_model, return_dict=True)

    def pad_id_change(self, ids):
        ids[ids == -100] = self.tokenizer.pad_token_id

    def train(self, epoch_num, alpha):
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
            outputs, gate_outputs = self.model(batch)
            logits = outputs.logits
            batch["label"]["input_ids"] = batch["label"].input_ids.to(self.device)
            batch["gate_label"] = batch["gate_label"].to(self.device)

            loss = self.loss_fn(
                logits.view(-1, self.tokenizer.vocab_size),
                batch["label"].input_ids.view(-1).to(self.device),
            )
            loss = loss.view(len(batch["gate_label"]), -1).mean(dim=1)
            predict_loss = (loss * batch["gate_label"]).sum(dim=0) / batch[
                "gate_label"
            ].sum(dim=0)
            gate_loss = self.gate_loss_fn(
                gate_outputs, batch["gate_label"].to(torch.float).reshape(-1, 1)
            )

            # Gate Loss
            if predict_loss.isnan():
                final_loss = gate_loss
            else:
                final_loss = alpha * predict_loss + (1 - alpha) * gate_loss

            final_loss.backward()
            loss_sum += final_loss.detach().item()

            self.optimizer.step()
            self.scheduler.step()
            if (iter + 1) % 10 == 0:

                self.logger.info(
                    f"Epoch {epoch_num} training : {iter+1}/{train_max_iter} loss : {loss_sum/10:.4f} lr:{self.optimizer.param_groups[0]['lr']:.6f}"
                )
                loss_sum = 0

                if (iter + 1) % 50 == 0:

                    outputs_idx = torch.argmax(outputs.logits, 2)

                    predict_text = self.tokenizer.batch_decode(
                        outputs_idx, skip_special_tokens=True
                    )
                    self.pad_id_change(batch["label"]["input_ids"])
                    answer_text = self.tokenizer.batch_decode(
                        batch["label"]["input_ids"], skip_special_tokens=True
                    )

                    pred_gate = torch.round(gate_outputs.view(-1)).detach().cpu()
                    answer_gate = batch["gate_label"].detach().cpu()

                    for (g_a, g_p, t_a, t_p) in zip(
                        answer_gate, pred_gate, answer_text, predict_text
                    ):
                        self.logger.info(
                            f"g_a {int(g_a.item())}, g_p {int(g_p.item())}"
                        )
                        if g_a == 1:
                            self.logger.info(f"t_a {t_a} t_p {t_p}")

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
                outputs, gate_outputs = self.model(batch)
                loss = outputs.loss.mean()

                loss_sum += loss.detach()
                if (iter + 1) % 50 == 0:
                    self.logger.info(
                        f"Epoch {epoch_num} Validation : {iter+1}/{valid_max_iter}"
                    )

        self.logger.info(f"Epoch {epoch_num} Validation loss : {loss_sum/iter:.4f}")
        return loss_sum / iter

    def test(self):
        answer, pred, answer_gate_list, pred_gate_list = [], [], [], []
        test_max_iter = int(len(self.test_data) / self.test_batch_size)
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_data,
            batch_size=self.test_batch_size,
            collate_fn=self.test_data.collate_fn,
        )
        pred_dict = defaultdict(
            lambda: defaultdict(dict)
        )  # dial_id, # turn_id # schema
        self.model.eval()
        loss_sum = 0
        self.logger.info("Test start")
        with torch.no_grad():
            for iter, batch in enumerate(test_loader):
                if iter % 10 == 0:
                    self.logger.info(f"Test... {iter+1}/{test_max_iter}")
                outputs, gate_outputs = self.model(batch)
                outputs_idx = torch.argmax(outputs.logits, 2)

                predict_text = self.tokenizer.batch_decode(
                    outputs_idx, skip_special_tokens=True
                )

                self.pad_id_change(batch["label"]["input_ids"])
                answer_text = self.tokenizer.batch_decode(
                    batch["label"]["input_ids"], skip_special_tokens=True
                )

                pred_gate = torch.round(gate_outputs.view(-1)).detach().cpu().tolist()
                answer_gate = batch["gate_label"].detach().cpu().tolist()

                answer.extend(answer_text)
                pred.extend(predict_text)
                answer_gate_list.extend(answer_gate)
                pred_gate_list.extend(pred_gate)

                for d, t, s, p_g, p_t in zip(
                    batch["dial_id"],
                    batch["turn_id"],
                    batch["schema"],
                    pred_gate,
                    predict_text,
                ):
                    if p_g == 1:
                        pred_dict[d][t][s] = p_t

        pred_dict = self.curr_belief_to_normal_belef(
            dict(pred_dict)
        )  # Previsou has only currrent belief

        return answer, pred, answer_gate_list, pred_gate_list, pred_dict

    def curr_belief_to_normal_belef(self, pred_dict):
        for d_id in pred_dict:
            for t_id in pred_dict[d_id]:
                if t_id == 0:
                    pass
                else:
                    pred_dict[d_id][t_id] = {
                        **pred_dict[d_id][t_id],
                        **pred_dict[d_id][t_id - 1],
                    }
        return pred_dict

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

    def save_test_result(self, pred_dict, test_data_path, save_path):

        test_dataset = json.load(open(test_data_path, "r"))

        for d_id in test_dataset.keys():
            dial = test_dataset[d_id]["log"]
            for turn in dial:
                t_id = int(turn["turn_num"])
                try:
                    turn.update({"pred": pred_dict[d_id][t_id]})
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
