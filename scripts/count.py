#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
from math import log
import math
import numpy as np

def main():
    """ main func
    """
    lb_counts = {}
    sum_ = 0
    n = 0
    with open('train', 'r', encoding='utf8') as ofile:
        for line in ofile:
            infos = line.strip().split("\t")
            n += 1
            for lb in infos[0].split("\001"):
                if lb not in lb_counts:
                    lb_counts[lb] = 0
                lb_counts[lb] += 1
                sum_ += 1

    lb_counts = sorted(iter(lb_counts.items()), key=lambda x:x[1], reverse=True)
    for k, v in lb_counts:
        idfe = log(n+1.0) - log(v+1.0)
        idf10 = math.log(n+1.0, 10) - math.log(v+1.0, 10)
        idf32 = math.log(n+1.0, 32) - math.log(v+1.0, 32)
        print("%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f" %(k, v, np.clip(n/v, 1, 3000), idfe, idf10, idf32))
            

if __name__ == '__main__':
    main()

#/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
