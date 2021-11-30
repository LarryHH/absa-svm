# SYSTEM
import os
import sys
import time
import re
import random
import configparser
from typing import Union, Tuple, List
from collections import defaultdict
from pathlib import Path
import pickle
from ast import literal_eval

# ML
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import scale

# NLP
import stanza
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('sentiwordnet')
# nltk.download('wordnet')

# MODULES
from dataset_depparse import *
import search_feature_comb

'''
ABSA-SVM TASK
1.0. Process dataset
1.1. Perform aspect and word clustering
1.2. Perform Chi-squared feature importance
2.0. Perform SVM classification
2.1. Feature selection
2.2. Statistics/performance
'''
DATA_ARGS = None
CLASSIFIERS = None
FEATURES = None
STOP_WORDS = None

def get_dataset():
    if DATA_ARGS.getboolean('depparse'):
        from dataset_depparse import Dataset
    else:
        from dataset import Dataset
    base_dir = DATA_ARGS['base_dir']
    proc = DATA_ARGS.getboolean('processed')
    bert = DATA_ARGS.getboolean('bert')
    ns = DATA_ARGS.getboolean('never_split')
    asp_c = DATA_ARGS.getint('n_aspect_clusters')
    word_c = DATA_ARGS.getint('n_word_clusters')
    print(f"Dataset params: {base_dir, proc, bert, ns, asp_c, word_c}")
    data = Dataset(base_dir=base_dir, is_preprocessed=proc, ns=ns, bert=bert, aspect_clusters=asp_c, word_clusters=word_c)
    return data

def load_config(fp):
    config = configparser.ConfigParser(allow_no_value=True)	
    config.optionxform = str	
    config.read(fp)
    data_args = config['DATA']
    clfs = config['CLASSIFIERS']
    features = config['FEATURES']

    features = feature_presets(features)

    classifiers = {}
    for clf in clfs:
        if clfs.getboolean(clf):
            classifiers[clf] = {}
            for k, v in dict(config[clf]).items():
                try:
                    v_eval = literal_eval(v) if v not in ['true', 'false'] else literal_eval(v.capitalize())
                except ValueError as e:
                    v_eval = v
                classifiers[clf][k] = v_eval
    print(classifiers)
    return data_args, classifiers, features

def feature_presets(features):
    if features.getboolean('use_smote_subsampling'):
        features.set('use_subsampling') == 'false'

def main():
    if not DATA_ARGS.getboolean('processed'):
        data = get_dataset()
    
    if CLASSIFIERS:
        for clf, args in CLASSIFIERS.items():
            search_feature_comb.main(DATA_ARGS, FEATURES, clf, args)

if __name__ == "__main__":

    DATA_ARGS, CLASSIFIERS, FEATURES = load_config('config/config.ini')
    main()