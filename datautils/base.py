#!/usr/bin/env python
# coding:utf8


import os
import torch.utils.data as data

def truncate_seq_pair(max_seq_len, tokens_a, tokens_b):
    """Truncate the sequence of pair, the last token will be removed
    if it is longer than the other.
    Args:
        max_seq_len: max length of target sequence
        tokens_a: first token sequence of the pair
        tokens_b: second token sequence of the pair
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq_len - 2: # for [CLS] and [SEP] tokens
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def find_first_sublist(main_list, sub_list):
    """Find the start and end indexes of sublist in main list
    Args:
        main_list: the main list.
        sub_list: the sublist.
    Return:
        the start and end indexes of sub_list in main_list
    """
    sub_len = len(sub_list)
    for i, _ in enumerate(main_list):
        if main_list[i: i + sub_len] == sub_list:
            return (i, i + sub_len - 1)


def prepare_files_offset(path, files_list, offset_list):
    """Fill the file index and offsets of each line in files_list in offset_list
    Args:
        path: string of file path, support single file or file dir
        files_list: the list contains file names
        offset_list: the list contains the tuple of file name index and offset
    """
    if os.path.isdir(path): # for multi-file, its input is a dir
        files_list.extend([os.path.join(path, f) for f in os.listdir(path)])
    elif os.path.isfile(path): # for single file, its input is a file
        files_list.append(path)
    else:
        raise RuntimeError(path + " is not a normal file.")
    for i, f in enumerate(files_list):
        offset = 0
        with open(f, "r", encoding="utf-8") as single_file:
            for line in single_file:
                tup = (i, offset)
                offset_list.append(tup)
                offset += len(bytes(line, encoding='utf-8'))


class Dataset(data.Dataset):
    """The base class of different task dataset
    """
    def __init__(self, training_path=None, testing_path=None):
        self.training_path = training_path
        self.testing_path = testing_path
        self.training_files = list()
        self.testing_files = list()
        self.training_files_offset = list()
        self.testing_files_offset = list()
        self.training_len = 0
        self.testing_len = 0
        self.is_training = True if training_path is not None else False
        self._check_files()

    def _check_files(self):
        if self.training_path is None and self.testing_path is None:
            raise RuntimeError("Training path and Testing path cannot be \
                empty at same time.")

        if self.training_path:
            if not os.path.exists(self.training_path):
                raise RuntimeError("Training files does not exist at " + \
                    self.training_path)
            prepare_files_offset(self.training_path, self.training_files,
                                 self.training_files_offset)
            self.training_len = len(self.training_files_offset)

        if self.testing_path:
            if not os.path.exists(self.testing_path):
                raise RuntimeError("Testing files does not exist at " + \
                    self.testing_path)
            prepare_files_offset(self.testing_path, self.testing_files,
                                 self.testing_files_offset)
            self.testing_len = len(self.testing_files_offset)

    def __len__(self):
        if self.is_training:
            return self.training_len
        return self.testing_len

    def _get_line(self, index):
        tup = self.training_files_offset[index] if self.is_training else \
            self.testing_files_offset[index]
        target_file = self.training_files[tup[0]] if self.is_training else \
            self.testing_files[tup[0]]
        line = ""
        with open(target_file, "r", encoding="utf-8") as f:
            f.seek(tup[1])
            line = f.readline()
        return line

class Collate(object):
    def __init__(self, dataset, tag_padding=0):
        self.full_name = dataset.full_name
        self.padding = dataset.tokenizer.vocab["[PAD]"]
        self.tag_padding = tag_padding

    def __call__(self, batch):
        raise NotImplementedError("This method need to be implemented")
