import os
import pdb
import pickle
import ontology
import csv, json
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score


# these are from trade-dst, https://github.com/jasonwu0731/trade-dst
import os
import csv, json

import logging
import ontology
from collections import defaultdict
import pdb

logger = logging.getLogger("my")
import pickle


def idx_to_text(tokenizer, idx):
    pass


def dict_to_csv(data, file_name):
    w = csv.writer(open(f"./logs/csvs/{file_name}", "w"))
    for k, v in data.items():
        w.writerow([k, v])
    w.writerow(["===============", "================="])


def dict_to_json(data, file_name):
    with open(f"./logs/jsons/{file_name}", "w") as fp:
        json.dump(data, fp, indent=4)


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def evaluate_metrics(all_prediction, raw_file_path, detail_log=True):
    schema = ontology.QA["all_domain"]
    domain = ontology.QA["bigger_domain"]

    detail_wrongs = defaultdict(lambda: defaultdict(list))
    turn_acc, joint_acc, turn_cnt, f1_cnt, f1 = 0, 0, 0, 0, 0
    schema_acc = {s: 0 for s in schema}
    domain_acc = {s: 0 for s in domain}
    raw_file = json.load(open(raw_file_path, "r"))
    for key in raw_file.keys():
        if key not in all_prediction.keys():
            continue
        dial = raw_file[key]["log"]
        for turn_idx, turn in enumerate(dial):
            belief_label = turn["belief"]
            try:
                belief_pred = all_prediction[key][turn_idx]
                # belief_pred =  turn['belief']
            except:
                if turn_idx == 0:
                    print("use answer")
                belief_pred = turn["belief"]

            belief_label = [f"{k} : {v}" for (k, v) in belief_label.items()]
            belief_pred = [f"{k} : {v}" for (k, v) in belief_pred.items()]
            if turn_idx == len(dial) - 1:
                logger.info(key)
                logger.info(f"label : {sorted(belief_label)}")
                logger.info(f"pred : {sorted(belief_pred)}")

            if set(belief_label) == set(belief_pred):
                joint_acc += 1

            ACC, schema_acc_temp, domain_acc_temp, detail_wrong = compute_acc(
                belief_label, belief_pred, schema, domain, detail_log
            )

            turn_acc += ACC
            schema_acc = {k: v + schema_acc_temp[k] for (k, v) in schema_acc.items()}
            domain_acc = {k: v + domain_acc_temp[k] for (k, v) in domain_acc.items()}
            f1_ = cal_f1(belief_label, belief_pred)
            if f1_ != None:
                f1_cnt += 1
                f1 += f1_

            if detail_log:
                detail_wrongs[key][turn_idx] = detail_wrong

            turn_cnt += 1

    domain_acc = {k: v / turn_cnt for (k, v) in domain_acc.items()}
    schema_acc = {k: v / turn_cnt for (k, v) in schema_acc.items()}
    joint_acc = joint_acc / turn_cnt
    turn_acc = turn_acc / turn_cnt
    f1 = f1 / f1_cnt

    return joint_acc, turn_acc, domain_acc, schema_acc, f1, detail_wrongs


def compute_acc(gold, pred, slot_temp, domain, detail_log):
    # import pdb; pdb.set_trace()
    detail_wrong = []
    miss_gold = 0
    wrong_pred = 0
    miss_slot = []
    schema_acc = {s: 1 for s in slot_temp}
    domain_acc = {s: 1 for s in domain}

    for g in gold:
        if g not in pred:
            miss_gold += 1
            schema_acc[g.split(" : ")[0]] -= 1
            if domain_acc[g.split("-")[0]] == 1:
                domain_acc[g.split("-")[0]] = 0
            miss_slot.append(g.split(" : ")[0])
            if detail_log:
                for p in pred:
                    if p.startswith(miss_slot[-1]):
                        detail_wrong.append((g, p))
                        break
                else:
                    detail_wrong.append((g, ontology.QA["NOT_MENTIONED"]))
    for p in pred:
        if p not in gold and p.split(" : ")[0] not in miss_slot:
            wrong_pred += 1
            schema_acc[p.split(" : ")[0]] -= 1
            if domain_acc[p.split("-")[0]] == 1:
                domain_acc[p.split("-")[0]] = 0
            if detail_log:
                detail_wrong.append((ontology.QA["NOT_MENTIONED"], p))

    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC, schema_acc, domain_acc, detail_wrong


def save_pickle(file_name, data):
    with open(file_name, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


def acc_metric(gold, pred):
    return accuracy_score(gold, pred)


def jga_metric(raw_file, all_prediction):
    joint_acc, turn_acc, turn_cnt = 0, 0, 0

    for key in raw_file.keys():
        if key not in all_prediction.keys():
            continue
        dial = raw_file[key]["log"]
        for turn_idx, turn in enumerate(dial):
            try:
                raw_belief_label = turn["belief"]
                raw_belief_pred = all_prediction[key][turn_idx]
                # if turn_idx != 0:
                # raw_belief_pred  = dict(all_prediction[key][turn_idx-1], **all_prediction[key][turn_idx])
            except:
                pdb.set_trace()
            belief_label = [f"{k} : {str(v)}" for (k, v) in raw_belief_label.items()]
            belief_pred = [f"{k} : {str(v)}" for (k, v) in raw_belief_pred.items()]

            if set(belief_label) == set(belief_pred):
                joint_acc += 1
            turn_cnt += 1

    return (
        joint_acc / turn_cnt,
        turn_acc / turn_cnt,
    )


def save_pickle(file_name, data):
    with open(file_name, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


# def compute_acc(gold, pred, slot_temp):
#     detail_wrong = []
#     miss_gold = 0
#     miss_slot = []
#     for g in gold:
#         if g not in pred:
#             miss_gold += 1
#             miss_slot.append(g.split(" : ")[0])
#     wrong_pred = 0
#     for p in pred:
#         if p not in gold and p.split(" : ")[0] not in miss_slot:
#             wrong_pred += 1
#     ACC_TOTAL = len(slot_temp)
#     ACC = len(slot_temp) - miss_gold - wrong_pred
#     ACC = ACC / float(ACC_TOTAL)
#     return ACC


def cal_num_same(a, p):
    answer_tokens = a
    pred_tokens = p
    common = Counter(answer_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    return num_same


def cal_f1(answer_tokens, pred_tokens, return_all=False):
    if len(answer_tokens) == 0:
        return None
    common = Counter(answer_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        precision = 0
        recall = 0
        mini_f1 = 0
    else:
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(answer_tokens)
        mini_f1 = (2 * precision * recall) / (precision + recall)
    if return_all:
        return precision, recall, mini_f1
    else:
        return mini_f1


if __name__ == "__main__":
    seed = 2
    # pred_file = json.load(open(f'logs/baseline_sample{seed}/pred_belief.json', "r"))
    pred_file = json.load(open(f"logs/baseline_sample1.0/pred_belief.json", "r"))

    ans_file = json.load(open("../woz-data/MultiWOZ_2.1/test_data.json", "r"))
    # unseen_data = json.load(open(f'../woz-data/MultiWOZ_2.1/split0.1/unseen_data{seed}0.1.json' , "r"))
    unseen_data = json.load(open(f"../woz-data/MultiWOZ_2.1/unseen_data.json", "r"))

    JGA, slot_acc, unseen_recall = evaluate_metrics(pred_file, ans_file, unseen_data)
    print(
        f"JGA : {JGA*100:.4f} ,  slot_acc : {slot_acc*100:.4f}, unseen_recall : {unseen_recall*100:.4f}"
    )
