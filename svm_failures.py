import os
from typing import Counter
from dataset_depparse import *
from file_utils import *
from collections import Counter
from ast import literal_eval
import re

NUMBERS = re.compile(r'(\d+)')

def numericalSort(value):
    parts = NUMBERS.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_indexes(fp):
    res = {
        'indexes': [],
        'f1': []
    }

    for fn in sorted(os.listdir(fp), key=numericalSort):
        with open(os.path.join(fp, fn), 'r') as f:
            lines = f.read().splitlines()
            indexes = literal_eval(lines[-1])
            res['indexes'].append(indexes)
            for line in lines:
                if 'macro_f1' in line:
                    f_score = float(line.split(':')[1].strip())
                    res['f1'].append(f_score)
    return res

def failures_indexes(data, res): # per cluster
    for i, (f1, index) in enumerate(zip(res['f1'], res['indexes'])):
        print(f'--- SVM: {i} ---')
        incorrect = [data.test_data[i] for i in index]

        # aspects
        aspects = Counter(s.aspect for s in incorrect)
        
        # polarity
        polarity = Counter(s.polarity for s in incorrect)
        new_keys = {-1: 'NEG', 0: 'NET', 1: 'POS'}                
        polarity = dict((new_keys[key], value) for (key, value) in polarity.items())
        polarity = dict(sorted(polarity.items(), key=lambda item: item[1], reverse=True))

        # print(aspects)
        print(f1, polarity) 


if __name__ == "__main__":
    base_dir = 'datasets/rest'
    res_fn = 'r500_k20_BERT'
    res_dir = f'{base_dir}/optimal_results/{res_fn}'

    data = Dataset(base_dir=base_dir, is_preprocessed=True)
    res = load_indexes(res_dir)
    failures_indexes(data, res)


