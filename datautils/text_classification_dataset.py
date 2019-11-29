#!/usr/bin/env python
# coding:utf8

import torch
from datautils.base import Dataset, Collate
from huggingface import tokenization
import numpy as np

def text_classification_collate_fn(batch, full_name, padding):
    """The collate function for text classification dataset.
    Args:
        batch: the list of different instance
        full_name: full name of the task
        padding: padding token index
    Return:
        The dict contains bert inputs and labels for text classification, each
        element is batched.
    """
    tokens = [instance["tokens"] for instance in batch]
    segment_ids = [[0] * len(token) for token in tokens]
    attn_masks = [[1] * len(token) for token in tokens]
    labels = [instance["label"] for instance in batch]
    max_len = max([len(token) for token in tokens])
    for i, token in enumerate(tokens):
        token.extend([padding] * (max_len - len(token)))
        segment_ids[i].extend([0] * (max_len - len(segment_ids[i])))
        attn_masks[i].extend([0] * (max_len - len(attn_masks[i])))
    tokens = torch.LongTensor(tokens)
    segment_ids = torch.LongTensor(segment_ids)
    attn_masks = torch.LongTensor(attn_masks)
    labels = torch.LongTensor(labels)
    return {"full_name": full_name, "tokens": tokens, "segment_ids": segment_ids,
            "attn_masks": attn_masks, "labels": labels}


class TextClassificationCollate(Collate):
    def __call__(self, batch):
        return text_classification_collate_fn(batch, self.full_name, self.padding)

class TextClassificationDataset(Dataset):
    def __init__(self, vocab_file, label_file, max_seq_len, do_lower_case=True,
                 resource="", *inputs, **kwargs):
        super(TextClassificationDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        import utils
        self.label2idx = utils.load_dict_from_file(label_file)
        self.idx2label = dict((v, k) for k, v in self.label2idx.items())
        self.max_seq_len = max_seq_len
        self.task_type = utils.TaskType.CLASSIFICATION
        self.full_name = resource + "_" + self.task_type

    def __getitem__(self, index):
        line = self._get_line(index)
        splits = line.strip().split("\t")
        tokens = self.tokenizer.tokenize(splits[1])
        tokens = tokens[:(self.max_seq_len - 1)]
        tokens = ["[CLS]"] + tokens
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return {"tokens": tokens, "label": self.label2idx[splits[0]]}


def multi_label_classification_collate_fn(batch, full_name, padding):
    """The collate function for text classification dataset.
    Args:
        batch: the list of different instance
        full_name: full name of the task
        padding: padding token index
    Return:
        The dict contains bert inputs and labels for text classification, each
        element is batched.
    """
    tokens = [instance["tokens"] for instance in batch]
    segment_ids = [[0] * len(token) for token in tokens]
    attn_masks = [[1] * len(token) for token in tokens]
    labels = [instance["label"] for instance in batch]
    labels_idx = [instance["label_idx"] for instance in batch]
    max_len = max([len(token) for token in tokens])
    contents = [instance["content"] for instance in batch]
    vids = [instance["vid"] for instance in batch]
    for i, token in enumerate(tokens):
        token.extend([padding] * (max_len - len(token)))
        segment_ids[i].extend([0] * (max_len - len(segment_ids[i])))
        attn_masks[i].extend([0] * (max_len - len(attn_masks[i])))
    tokens = torch.LongTensor(tokens)
    segment_ids = torch.LongTensor(segment_ids)
    attn_masks = torch.LongTensor(attn_masks)
    labels = torch.FloatTensor(labels)
    return {"full_name": full_name, "tokens": tokens, "segment_ids": segment_ids,
            "attn_masks": attn_masks, "labels": labels, 'labels_idx': labels_idx,
            "contents" : contents, "vids" : vids,}


class MultiLabelClassificationCollate(Collate):
    def __call__(self, batch):
        return multi_label_classification_collate_fn(batch, self.full_name, self.padding)

class MultiLabelClassificationDataset(Dataset):
    def __init__(self, vocab_file, label_file, max_seq_len, label_weight_file, do_lower_case=True, resource="", *inputs, **kwargs):
        super(MultiLabelClassificationDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
        import utils
        self.label2idx = utils.load_dict_from_file(label_file)
        self.idx2label = dict((v, k) for k, v in self.label2idx.items())
        self.max_seq_len = max_seq_len
        self.task_type = utils.TaskType.CLASSIFICATION
        self.full_name = resource + "_" + self.task_type
        label_weight_info = utils.load_label_weight(label_weight_file)
        self.label_weight = np.ones(len(self.label2idx))
        for lb, idx in self.label2idx.items():
            self.label_weight[idx] = label_weight_info[lb]
        self.label_weight = torch.FloatTensor(self.label_weight)

    def __getitem__(self, index):
        """
        Data structure:
        label\tcontents\tvid
        """
        line = self._get_line(index)
        splits = line.strip().split("\t")
        content = splits[1]
        vid = splits[2]
        tokens = self.tokenizer.tokenize(splits[1])
        tokens = tokens[:(self.max_seq_len - 1)]
        tokens = ["[CLS]"] + tokens
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        try:
            label_idx = [self.label2idx[lb] for lb in splits[0].split('!;')]
        except:
            label_idx = [0]
        return {"tokens": tokens, "label": self.convert_labels(label_idx, len(self.label2idx)), 'label_idx': label_idx, 'content': content,'vid': vid}

    def convert_labels(self, labels, label_size):
        v = np.zeros(label_size)
        for i in labels:
            v[i] = 1
        return v