import torch
import numpy as np
import random

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def chunk(seq):
    offset = 512
    cur, cnt = 0, 0
    result = list()
    while cur < len(seq):
        result.append(seq[cur: cur + offset])
        cur += 450
        cnt += 1
    # print(result)
    return result, cnt

def eval(predictions, ground_truth, queLen=10):
    true_list = []
    with open(ground_truth, 'r') as f:
        for i in f:
            name, label = i.split(',')
            true_list.append([j for j in label.split()])
    Map = 0
    for idx, i in enumerate(predictions):
        Map_temp = 0
        match = 0
        for j, k in enumerate(i):
            # print(idx, j)
            if k in true_list[idx]:
                match += 1
                Map_temp += (match / (j + 1))
        Map += Map_temp / min(len(true_list[idx]), 100)
    print("acc: {}".format(Map / queLen))