#!/usr/bin/env python
# coding:utf8

import torch
from datautils.base import Dataset, Collate
from datautils.base import find_first_sublist
from huggingface import tokenization

POINTWISE_COL_NUM = 4
PAIRWISE_COL_NUM = 5

def span_ranking_collate_fn(batch, full_name, padding):
    is_pairwise = batch[0]["is_pairwise"]
    seq_index = [each["seq_index"] for each in batch]
    labels = torch.LongTensor([each["label"] for each in batch])
    tokens = [each["tokens"] for each in batch]
    segment_ids = [[0] * len(each) for each in tokens]
    attn_masks = [[1] * len(each) for each in tokens]
    span_index = torch.LongTensor([each["span_index"] for each in batch])
    max_len = max([len(each) for each in tokens])
    for i, token in enumerate(tokens):
        token.extend([padding] * (max_len - len(token)))
        segment_ids[i].extend([0] * (max_len - len(segment_ids[i])))
        attn_masks[i].extend([0] * (max_len - len(attn_masks[i])))
    tokens = torch.LongTensor(tokens)
    segment_ids = torch.LongTensor(segment_ids)
    attn_masks = torch.LongTensor(attn_masks)
    span_index = torch.LongTensor(span_index)
    outputs_dict = {"is_pairwise" : is_pairwise, "seq_index" : seq_index,
                    "tokens" : tokens, "segment_ids" : segment_ids,
                    "attn_masks" : attn_masks, "span_index" : span_index,
                    "labels" : labels, "full_name" : full_name}
    if is_pairwise:
        second_span_index = torch.LongTensor([each["second_span_index"] for each in batch])
        outputs_dict["second_span_index"] = second_span_index
    return outputs_dict


class SpanRankingCollate(Collate):
    def __call__(self, batch):
        return span_ranking_collate_fn(batch, self.full_name, self.padding)


class SpanRankingDataset(Dataset):
    def __init__(self, vocab_file, max_seq_len, is_pairwise=False,
                 do_lower_case=True, resource="", *inputs, **kwargs):
        super(SpanRankingDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_seq_len = max_seq_len
        self.is_pairwise = is_pairwise
        import utils
        self.task_type = utils.TaskType.SPANRANKING
        self.full_name = resource + "_" + self.task_type

    def _pointwise_process(self, index):
        """
        Data structure:
        content_id\tcontent\talternative_sequence\tlabel
        Reference: ./toy_data/text_matching/sample.train.pointwise
        """
        line = self._get_line(index)
        splits = line.strip().split("\t")
        if len(splits) < POINTWISE_COL_NUM:
            raise RuntimeError("Data is not illegal: " + line)
        seq_index = splits[0]
        tokens_a = self.tokenizer.tokenize(splits[1])
        tokens_b = self.tokenizer.tokenize(splits[2])
        tokens_a = tokens_a[:(self.max_seq_len - 1)]
        tokens_a = ["[CLS]"] + tokens_a
        tokens_a_idx = self.tokenizer.convert_tokens_to_ids(tokens_a)
        tokens_b_idx = self.tokenizer.convert_tokens_to_ids(tokens_b)
        span_index = find_first_sublist(tokens_a_idx, tokens_b_idx)
        if span_index is None:
            raise RuntimeError("Cannot find the span in: " + line)
        return {"is_pairwise" : False, "seq_index" : seq_index,
                "tokens" : tokens_a_idx, "span_index" : span_index,
                "label" : int(splits[3])}

    def _pairwise_process(self, index):
        """
        Data structure:
        content_id\tcontent\tfirst_alternative_sequence\tsecond_alternative_sequence\tlabel
        Reference: ./toy_data/text_matching/sample.train.pairwise
        """
        line = self._get_line(index)
        splits = line.strip().split("\t")
        if len(splits) < PAIRWISE_COL_NUM:
            raise RuntimeError("Data is not illegal: " + line)
        seq_index = splits[0]
        tokens_a = self.tokenizer.tokenize(splits[1])
        tokens_b = self.tokenizer.tokenize(splits[2])
        tokens_c = self.tokenizer.tokenize(splits[3])
        tokens_a = tokens_a[:(self.max_seq_len - 1)]
        tokens_a = ["[CLS]"] + tokens_a
        tokens_a_idx = self.tokenizer.convert_tokens_to_ids(tokens_a)
        tokens_b_idx = self.tokenizer.convert_tokens_to_ids(tokens_b)
        tokens_c_idx = self.tokenizer.convert_tokens_to_ids(tokens_c)
        span_index = find_first_sublist(tokens_a_idx, tokens_b_idx)
        second_span_index = find_first_sublist(tokens_a_idx, tokens_c_idx)
        if span_index is None or second_span_index is None:
            raise RuntimeError("Cannot find the span in: " + line)
        return {"is_pairwise" : True, "seq_index" : seq_index,
                "tokens" : tokens_a_idx, "span_index" : span_index,
                "second_span_index" : second_span_index, "label" : int(splits[4])}

    def __getitem__(self, index):
        if self.is_training and self.is_pairwise:
            return self._pairwise_process(index)
        return self._pointwise_process(index)