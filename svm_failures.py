from curses import meta
import os
from typing import Counter
from dataset_depparse import *
from file_utils import *
from collections import Counter
from ast import literal_eval
import re
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import json


NUMBERS = re.compile(r'(\d+)')

def numericalSort(value):
    parts = NUMBERS.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def load_indexes(fp):
    res = {
        'cluster': [],
        'indexes': [],
        'f1': [],
        'preds': []
    }
    filenames = os.listdir(fp)
    filenames.remove('config.ini')
    filenames = [f for f in filenames if target in f]
    filenames = sorted(filenames, key=numericalSort)
    
    for fn in filenames:
        with open(os.path.join(fp, fn), 'r') as f:
            lines = f.read().splitlines()
            indexes = literal_eval(lines[-3])
            preds = literal_eval(lines[-1])
            res['indexes'].append(indexes)
            res['preds'].append(preds)
            res['cluster'].append([int(s) for s in fn.split('_') if s.isdigit()][0])
            for line in lines:
                if 'macro_f1' in line:
                    f_score = float(line.split(':')[1].strip())
                    res['f1'].append(f_score)
    return res

def to_csv_from_record(record, path, metadata):
    df = pd.DataFrame.from_records(record)
    with open(path, 'w') as f:
        if metadata:
            f.write(json.dumps(metadata) + '\n')       
        df.to_csv(f, index=False)

def cluster_investigation(data, res, avg_acc, avg_f1): # per cluster
    for i, f1, index, pred in zip(res['cluster'], res['f1'], res['indexes'], res['preds']):
        if f1 < avg_f1:

            train_instances = [s.to_dict() for s in data.train_data if s.aspect_cluster == i]
            test_instances = [s.to_dict() for s in data.test_data if s.aspect_cluster == i]

            print(f'--- SVM: {i} --- F1: {f1}, #FAILS: {len(index)}/{len(test_instances)}')

            incorrect = [data.test_data[ii].to_dict() for ii in index]
            polarity = Counter(s['polarity'] for s in incorrect)
            new_keys = {-1: 'NEG', 0: 'NET', 1: 'POS'}                
            polarity = dict((new_keys[key], value) for (key, value) in polarity.items())
            polarity = dict(sorted(polarity.items(), key=lambda item: item[1], reverse=True))
            total_polarity = dict((new_keys[key], value) for (key, value) in Counter(s['polarity'] for s in test_instances).items())

            for (sample, p) in zip(incorrect, pred):
                sample['pred_polarity'] = p

            metadata = {
                'f1': f1,
                'acc': (len(test_instances) - len(index)) / len(test_instances),
                'avg_f1': avg_f1,
                'avg_acc': avg_acc,
                'fails': len(index),
                'total_test_instances': len(test_instances),
                'total_polarity': total_polarity,
                'incorrect_polarity': polarity
            }

            Path(f"{base_dir}/analysis/{res_fn}/cluster_{i}/").mkdir(parents=True, exist_ok=True)

            to_csv_from_record(test_instances, f"{base_dir}/analysis/{res_fn}/cluster_{i}/test_instances.csv", None)
            to_csv_from_record(train_instances, f"{base_dir}/analysis/{res_fn}/cluster_{i}/train_instances.csv", None)
            to_csv_from_record(incorrect, f"{base_dir}/analysis/{res_fn}/cluster_{i}/incorrect.csv", metadata)



def failures_indexes(data, res, avg_acc, avg_f1): # per cluster
    for i, (f1, index, pred) in enumerate(zip(res['f1'], res['indexes'], res['preds'])):
        print(f'--- SVM: {i} --- F1: {f1}, #FAILS: {len(index)}')
        incorrect = [data.test_data[i] for i in index]
        
        polarity = Counter(s.polarity for s in incorrect)
        new_keys = {-1: 'NEG', 0: 'NET', 1: 'POS'}                
        polarity = dict((new_keys[key], value) for (key, value) in polarity.items())
        polarity = dict(sorted(polarity.items(), key=lambda item: item[1], reverse=True))

        print(f'--- {polarity}')
        for (sample, p) in zip(incorrect, pred):
            print(f"ID: {sample.id}, T_LABEL: {new_keys[sample.polarity]}, P_LABEL: {new_keys[p]}, ASPECT: {sample.aspect}")
            print(f"TEXT: {sample.text}")

def numerical_sort(value):
    parts = NUMBERS.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def cal_acc(result_dir, clf):
    filenames = os.listdir(result_dir)
    result_dict = {}
    filenames = [f for f in filenames if f.split('_')[0] == clf]
    if not filenames:
        return None 
    for fn in sorted(filenames, key=numerical_sort):
        final_path = os.path.join(result_dir, fn)
        with open(final_path, 'r') as f:
            for line in f.readlines():
                if 'correct / total' in line:
                    tmp = line.split(':')[1]
                    correct = int(tmp.split('/')[0].strip())
                    total = int(tmp.split('/')[1].strip())
                    if 'correct' not in result_dict.keys():
                        result_dict['correct'] = [correct]
                    else:
                        result_dict['correct'].append(correct)
                    if 'total' not in result_dict.keys():
                        result_dict['total'] = [total]
                    else:
                        result_dict['total'].append(total)
                if 'macro_f1' in line:
                    f_score = float(line.split(':')[1].strip())
                    if 'f1' not in result_dict.keys():
                        result_dict['f1'] = [f_score]
                    else:
                        result_dict['f1'].append(f_score)

    correct = sum(result_dict['correct'])
    total = sum(result_dict['total'])
    f_scores = result_dict['f1']
    f1 = 0
    for num_sample, chunk_f in zip(result_dict['total'], f_scores):
        f1 += num_sample / total * chunk_f
    return correct, total, f1

def remake_dir(path):
    dirpath = Path(path)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    base_dir = 'datasets/rest'
    res_fn = 'r2000_k50_BERT'
    res_dir = f'{base_dir}/optimal_results/{res_fn}'
    target = 'SVC'

    correct, total, f1 = cal_acc(res_dir, target)
    acc = correct/total    

    print(acc, f1)

    data = Dataset(base_dir=base_dir, is_preprocessed=True)
    res = load_indexes(res_dir)
    # failures_indexes(data, res, acc, f1)

    remake_dir(f"{base_dir}/analysis/{res_fn}/")
    cluster_investigation(data, res, acc, f1)
