#!/usr/bin/env python
# coding:utf8
# Provide some utilities for task finetune scripts.

import collections
import json
import os
import torch
import torch.utils.data as data
from datautils.matching_dataset import MatchingDataset, MatchingCollate
from datautils.span_ranking_dataset import SpanRankingDataset, SpanRankingCollate
from datautils.tagging_dataset import TaggingDataset, TaggingCollate
from datautils.text_classification_dataset import TextClassificationDataset
from datautils.text_classification_dataset import TextClassificationCollate
from huggingface.optimization import BERTAdam

EPSILON = 1e-8

class TaskInfo(object):
    def __init__(self, task_type, resource="", label_file=None, training_path=None,
                 testing_path=None, is_pairwise=False):
        self.resource = resource
        if task_type not in [TaskType.CLASSIFICATION, TaskType.MATCHING, \
            TaskType.SPANRANKING] and task_type not in TaskType.TAGGING:
            raise RuntimeError(task_type + " is not supported.")
        self.task_type = task_type
        self.full_key = self.resource + "_" + self.task_type
        self.label_file = label_file
        self.num_labels = 2
        if self.label_file is not None:
            self.num_labels = len(load_dict_from_file(self.label_file))
        self.training_path = training_path
        self.testing_path = testing_path
        self.is_pairwise = is_pairwise


class TaskType(object):
    CLASSIFICATION = "Classification"
    MATCHING = "Matching"
    TAGGING = ("Segmentation", "Postag")
    SPANRANKING = "Spanranking"


def load_dict_from_file(file_dir):
    """
    Load the dictionary from file, every line is a key info
    in file, and each line must at least have one word for KEY.
    The value is the index of KEY in file.
    """
    d = collections.OrderedDict()
    index = 0
    with open(file_dir, "r",encoding='utf-8') as f:
        for line in f:
            token = line.strip().split("\t")[0]
            d[token] = index
            index += 1
    return d


def load_json_config(config_path):
    with open(config_path, "r",encoding='utf-8') as f:
        config = json.load(f)
    return config


def load_label_weight(config_path):
    label2weight = {}
    with open(config_path, 'r',encoding='utf-8') as f:
        for line in f:
            infos = line.strip().split("\t")
            label2weight[infos[0]] = float(infos[2])
    
    return label2weight


def load_saved_model(model, saved_model_path):
    model_weight = torch.load(saved_model_path,map_location='cpu')
    new_state_dict = collections.OrderedDict()
    for k, v in model_weight.items():
        if k.startswith("module"):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


def init_bert_adam_optimizer(model, training_data_len, epoch, batch_size,
                             gradient_accumulation_steps, init_lr, warmup_proportion):
    no_decay = ["bias", "gamma", "beta"]
    optimizer_parameters = [
        {"params" : [p for name, p in model.named_parameters() \
            if name not in no_decay], "weight_decay_rate" : 0.01},
        {"params" : [p for name, p in model.named_parameters() \
            if name in no_decay], "weight_decay_rate" : 0.0}
    ]
    num_train_steps = int(training_data_len / batch_size / \
        gradient_accumulation_steps * epoch)
    optimizer = BERTAdam(optimizer_parameters,
                         lr=init_lr,
                         warmup=warmup_proportion,
                         t_total=num_train_steps)
    return optimizer


def save_model(experiment_name, model, epoch):
    if not os.path.exists(experiment_name):
        os.mkdir(experiment_name)
    model_weight = model.state_dict()
    new_state_dict = collections.OrderedDict()
    for k, v in model_weight.items():
        if k.startswith("module"):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model_name = "Epoch_" + str(epoch) + ".bin"
    model_dir = os.path.join(experiment_name, model_name)
    torch.save(new_state_dict, model_dir)


def init_multitask_task_infos(multitask_configs):
    task_infos = list()
    for task_config in multitask_configs:
        task_infos.append(TaskInfo(task_config.get("task_type"),
                                   task_config.get("resource", ""),
                                   task_config.get("label_file"),
                                   task_config.get("training_path"),
                                   task_config.get("testing_path"),
                                   bool(task_config.get("is_pairwise", 0))))
    return task_infos


def init_multitask_datasets(vocab_file, max_seq_len, task_infos):
    datasets = list()
    for task_info in task_infos:
        if task_info.task_type == TaskType.CLASSIFICATION:
            datasets.append(TextClassificationDataset(vocab_file=vocab_file,
                                                      label_file=task_info.label_file,
                                                      max_seq_len=max_seq_len,
                                                      resource=task_info.resource,
                                                      training_path=task_info.training_path,
                                                      testing_path=task_info.testing_path))
        elif task_info.task_type == TaskType.MATCHING:
            datasets.append(MatchingDataset(vocab_file=vocab_file,
                                            max_seq_len=max_seq_len,
                                            is_pairwise=task_info.is_pairwise,
                                            resource=task_info.resource,
                                            training_path=task_info.training_path,
                                            testing_path=task_info.testing_path))
        elif task_info.task_type == TaskType.SPANRANKING:
            datasets.append(SpanRankingDataset(vocab_file=vocab_file,
                                               max_seq_len=max_seq_len,
                                               is_pairwise=task_info.is_pairwise,
                                               resource=task_info.resource,
                                               training_path=task_info.training_path,
                                               testing_path=task_info.testing_path))
        elif task_info.task_type in TaskType.TAGGING:
            datasets.append(TaggingDataset(vocab_file=vocab_file,
                                           label_file=task_info.label_file,
                                           max_seq_len=max_seq_len,
                                           resource=task_info.resource,
                                           tagging_type=task_info.task_type,
                                           training_path=task_info.training_path,
                                           testing_path=task_info.testing_path))
    return datasets


def gen_multitask_dataloader(datasets, pin_memory, batch_size, num_workers, shuffle):
    dataloaders = list()
    for dataset in datasets:
        collate_fn = None
        if dataset.task_type == TaskType.CLASSIFICATION:
            collate_fn = TextClassificationCollate(dataset)
        elif dataset.task_type == TaskType.MATCHING:
            collate_fn = MatchingCollate(dataset)
        elif dataset.task_type == TaskType.SPANRANKING:
            collate_fn = SpanRankingCollate(dataset)
        elif dataset.task_type in TaskType.TAGGING:
            collate_fn = TaggingCollate(dataset)
        dataloader = data.DataLoader(dataset=dataset,
                                     collate_fn=collate_fn,
                                     pin_memory=pin_memory,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=shuffle)
        dataloaders.append(dataloader.__iter__())
    is_yield = [False] * len(dataloaders)
    while not all(is_yield):
        for i, dataloader in enumerate(dataloaders):
            if all(is_yield):
                break
            try:
                batch = dataloader.__next__()
                yield batch
            except StopIteration:
                is_yield[i] = True
                continue


def prepare_multitask_inputs(full_name, batch, margin, use_cuda, master_gpu_id):
    input_dicts = dict()
    tokens = batch.get("tokens")
    segment_ids = batch.get("segment_ids")
    attn_masks = batch.get("attn_masks")
    second_tokens = batch.get("second_tokens")
    second_segment_ids = batch.get("second_segment_ids")
    second_attn_masks = batch.get("second_attn_masks")
    labels = batch.get("labels")
    span_index = batch.get("span_index")
    second_span_index = batch.get("second_span_index")
    if use_cuda:
        tokens = tokens.cuda(master_gpu_id) if tokens is not None else None
        segment_ids = segment_ids.cuda(master_gpu_id) if segment_ids is not None else None
        attn_masks = attn_masks.cuda(master_gpu_id) if attn_masks is not None else None
        second_tokens = second_tokens.cuda(master_gpu_id) if second_tokens is not None else None
        second_segment_ids = second_segment_ids.cuda(master_gpu_id) if second_segment_ids \
            is not None else None
        second_attn_masks = second_attn_masks.cuda(master_gpu_id) if second_attn_masks \
            is not None else None
        labels = labels.cuda(master_gpu_id) if labels is not None else None
        span_index = span_index.cuda(master_gpu_id) if span_index is not None else None
        second_span_index = second_span_index.cuda(master_gpu_id) if second_span_index \
            is not None else None
    input_dicts["tokens"] = tokens
    input_dicts["segment_ids"] = segment_ids
    input_dicts["attn_masks"] = attn_masks
    input_dicts["second_tokens"] = second_tokens
    input_dicts["second_segment_ids"] = second_segment_ids
    input_dicts["second_attn_masks"] = second_attn_masks
    label_info = dict()
    label_info["labels"] = labels
    label_info["span_index"] = span_index
    label_info["second_span_index"] = second_span_index
    label_info["margin"] = margin
    input_dicts["task_labels"] = {full_name : label_info}
    return input_dicts


def logging_classification_info(logging, num_sample, correct_sum, evaluator,
                                p_list, r_list, fscore_list):
    logging.info("Total Testing Samples: " + str(num_sample))
    logging.info("Correct Prediction: " + str(correct_sum))
    logging.info("Error Rate: " + format(1 - (correct_sum / num_sample), "0.4f"))
    for i, _ in enumerate(p_list):
        logging.info("Level_" + str(i) + ", Precision: " + \
            format(p_list[i][evaluator.MICRO_AVERAGE],
                   "0.4f") + ", Recall: " + \
            format(r_list[i][evaluator.MICRO_AVERAGE],
                   "0.4f") + ", F1_score: " + \
            format(fscore_list[i][evaluator.MICRO_AVERAGE],
                   "0.4f"))


def logging_ranking_info(logging, ranking_prf, topk_list):
    logging.info("Ranking TopK Performance:")
    for i, prf in enumerate(ranking_prf["topk"][1]):
        logging.info("Top-" + str(topk_list[i]) + ", Precision: " + \
                     format(prf[0], "0.4f") + ", Recall: " + \
                     format(prf[1], "0.4f") + ", F1_score: " + \
                     format(prf[2], "0.4f") + ". Upperbound: " + \
                     format(ranking_prf["topk_upperbound"][1][i][0], "0.4f") + \
                     "//" + format(ranking_prf["topk_upperbound"][1][i][1], "0.4f") + \
                     "//" + format(ranking_prf["topk_upperbound"][1][i][2], "0.4f"))


def logging_tagging_info(logging, tagging_type, results):
    if tagging_type == "Segmentation":
        precision, recall, fscore = results
        logging.info("Precision: " + format(precision, "0.4f"))
        logging.info("Recall: " + format(recall, "0.4f"))
        logging.info("F1_score: " + format(fscore, "0.4f"))
    elif tagging_type == "Postag":
        accuracy = results
        logging.info("Accuracy: " + format(accuracy, "0.4f"))
