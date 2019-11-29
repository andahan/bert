#!/usr/bin/env python
# coding:utf8
# Convert npy model to pytorch model.


import collections
import os
import torch
import numpy as np


def convert(dump_path, npy_path):
    state_dict = collections.OrderedDict()
    for npy in os.listdir(npy_path):
        full_path = os.path.join(npy_path, npy)
        name = npy[:-4]
        params = np.load(full_path)
        state_dict[name] = torch.FloatTensor(params)
    torch.save(state_dict, os.path.join(dump_path, "ernie_for_hf.bin"))


if __name__ == "__main__":
    convert(os.sys.argv[1], os.sys.argv[2])
