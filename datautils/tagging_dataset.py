#!/usr/bin/env python
# coding:utf8


import torch
from datautils.base import Dataset, Collate
from huggingface import tokenization


TAGGING_COL_NUM = 1

def tagging_collate_fn(batch, full_name, token_padding, tag_padding):
    tokens = [each["tokens"] for each in batch]
    segment_ids = [[0] * len(each) for each in tokens]
    attn_masks = [[1] * len(each) for each in tokens]
    labels = [each["label"] for each in batch]
    labels_len = [len(each) for each in labels]
    token_max_len = max([len(each) for each in tokens])
    label_max_len = max([len(each) for each in labels])
    for i, token in enumerate(tokens):
        token.extend([token_padding] * (token_max_len - len(token)))
        segment_ids[i].extend([0] * (token_max_len - len(segment_ids[i])))
        attn_masks[i].extend([0] * (token_max_len - len(attn_masks[i])))
        labels[i].extend([tag_padding] * (label_max_len - len(labels[i])))
    tokens = torch.LongTensor(tokens)
    segment_ids = torch.LongTensor(segment_ids)
    attn_masks = torch.LongTensor(attn_masks)
    labels = torch.LongTensor(labels)
    return {"full_name" : full_name, "tokens" : tokens, "segment_ids" : segment_ids,
            "attn_masks" : attn_masks, "labels" : labels, "labels_len" : labels_len}


class TaggingCollate(Collate):
    def __call__(self, batch):
        return tagging_collate_fn(batch, self.full_name, self.padding, self.tag_padding)


class TaggingDataset(Dataset):
    def __init__(self, vocab_file, label_file, max_seq_len, do_lower_case=True,
                 resource="", tagging_type="", *inputs, **kwargs):
        super(TaggingDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_seq_len = max_seq_len
        import utils
        self.label2idx = utils.load_dict_from_file(label_file)
        self.task_type = tagging_type
        self.full_name = resource + "_" + self.task_type

    def __getitem__(self, index):
        line = self._get_line(index)
        splits = line.strip().split("\t")
        if len(splits) < TAGGING_COL_NUM:
            raise RuntimeError("Data is illegal: " + line)
        chars = list()
        tags = list()
        chars.append(self.tokenizer.vocab["[CLS]"])
        for pair in splits[0].split(" "):
            char, tag = pair.rsplit("/", 1)
            if len(chars) >= self.max_seq_len:
                break
            chars.append(self.tokenizer.vocab.get(char,
                                                  self.tokenizer.vocab["[UNK]"]))
            tags.append(self.label2idx[tag])
        return {"tokens" : chars, "label" : tags}