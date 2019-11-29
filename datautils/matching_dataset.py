#!/usr/bin/env python
# coding:utf8

import copy
import torch
from datautils.base import Dataset, Collate
from datautils.base import truncate_seq_pair
from huggingface import tokenization


POINTWISE_COL_NUM = 4
PAIRWISE_COL_NUM = 5


def get_input(token_name, segment_name, batch, padding):
    tokens = [each[token_name] for each in batch]
    segment_ids = [each[segment_name] for each in batch]
    attn_masks = [[1] * len(each) for each in tokens]
    max_len = max([len(each) for each in tokens])
    for i, token in enumerate(tokens):
        token.extend([padding] * (max_len - len(token)))
        segment_ids[i].extend([0] * (max_len - len(segment_ids[i])))
        attn_masks[i].extend([0] * (max_len - len(attn_masks[i])))
    tokens = torch.LongTensor(tokens)
    segment_ids = torch.LongTensor(segment_ids)
    attn_masks = torch.LongTensor(attn_masks)
    return tokens, segment_ids, attn_masks


def matching_collate_fn(batch, full_name, padding):
    is_pairwise = batch[0]["is_pairwise"]
    seq_index = [each["seq_index"] for each in batch]
    labels = torch.LongTensor([each["label"] for each in batch])
    outputs_dict = {"full_name" : full_name, "is_pairwise" : is_pairwise, "seq_index" : seq_index,
                    "labels" : labels}
    if is_pairwise:
        first_tokens, first_segment_ids, first_attn_masks = get_input("first_tokens",
                                                                      "first_segment_ids",
                                                                      batch,
                                                                      padding)
        second_tokens, second_segment_ids, second_attn_masks = get_input("second_tokens",
                                                                         "second_segment_ids",
                                                                         batch,
                                                                         padding)
        outputs_dict["tokens"] = first_tokens
        outputs_dict["segment_ids"] = first_segment_ids
        outputs_dict["attn_masks"] = first_attn_masks
        outputs_dict["second_tokens"] = second_tokens
        outputs_dict["second_segment_ids"] = second_segment_ids
        outputs_dict["second_attn_masks"] = second_attn_masks
    else:
        tokens, segment_ids, attn_masks = get_input("tokens", "segment_ids",
                                                    batch, padding)
        outputs_dict["tokens"] = tokens
        outputs_dict["segment_ids"] = segment_ids
        outputs_dict["attn_masks"] = attn_masks
    return outputs_dict


class MatchingCollate(Collate):
    def __call__(self, batch):
        return matching_collate_fn(batch, self.full_name, self.padding)


class MatchingDataset(Dataset):
    def __init__(self, vocab_file, max_seq_len, is_pairwise=False,
                 do_lower_case=True, resource="", *inputs, **kwargs):
        super(MatchingDataset, self).__init__(*inputs, **kwargs)
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case)
        self.max_seq_len = max_seq_len
        self.is_pairwise = is_pairwise
        import utils
        self.task_type = utils.TaskType.MATCHING
        self.full_name = resource + "_" + self.task_type

    def _text_process(self, seq_a, seq_b):
        tokens_a = self.tokenizer.tokenize(seq_a)
        tokens_b = self.tokenizer.tokenize(seq_b)
        truncate_seq_pair(self.max_seq_len, tokens_a, tokens_b)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * len(tokens_b)
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        return tokens, segment_ids

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
        tokens, segment_ids = self._text_process(splits[1], splits[2])
        return {"is_pairwise" : False, "seq_index" : seq_index, "tokens" : tokens,
                "segment_ids" : segment_ids, "label" : int(splits[3])}

    def _pairwise_process(self, index):
        """
        Data structure:
        content_id\tcontent\tfirst_alternative_sequence\tsecond_alternative_sequence\tlabel
        """
        line = self._get_line(index)
        splits = line.strip().split("\t")
        if len(splits) < PAIRWISE_COL_NUM:
            raise RuntimeError("Data is not illegal: " + line)
        seq_index = splits[0]
        first_content = splits[1]
        second_content = copy.deepcopy(first_content)
        first_tokens, first_segment_ids = self._text_process(first_content, splits[2])
        second_tokens, second_segment_ids = self._text_process(second_content, splits[3])
        return {"is_pairwise" : True, "seq_index" : seq_index,
                "first_tokens" : first_tokens, "second_tokens" : second_tokens,
                "first_segment_ids" : first_segment_ids, "second_segment_ids" : second_segment_ids,
                "label" : int(splits[4])}

    def __getitem__(self, index):
        if self.is_training and self.is_pairwise:
            return self._pairwise_process(index)
        return self._pointwise_process(index)
