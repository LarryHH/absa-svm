from dataset_depparse import *
from file_utils import *
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from lexicon_features import *
# from hyperopt_svm import HyperoptTuner
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import scale
from hyperopt_libsvm import HyperoptTunerLibSVM
import time
from pathlib import Path
import pickle

from transformers import BertTokenizerFast

REST_DIR = 'datasets/rest/'
LAPTOP_DIR = 'datastes/laptops/'
TOKENIZER = None
EMBED_DICT = None
stop_words = stop_words()

def get_aspect_embedding(sample):
    tokens = TOKENIZER.tokenize( "[CLS] " + sample.aspect + " [SEP]")
    tokens = [t for t in tokens if t != '[CLS]' and t != '[SEP]']
    embeddings = [v['embedding'] for t in tokens for k, v in EMBED_DICT[t].items() if v['sample_id'] == sample.id]
    if len(embeddings) > 1:
        embeddings = np.mean(embeddings, axis=0)
        return embeddings
    return embeddings[0]

def generate_vectors(train_data, test_data, bf, asp, lsa_k=None):
    if bf == 'all_words':
        x_train_tfidf, x_test_tfidf, x_train_pos_vec, x_test_pos_vec = dependent_features_vectors([s.words for s in train_data],
                                                                 [s.words for s in test_data],
                                                                 [s.pos_tags for s in train_data],
                                                                 [s.pos_tags for s in test_data])
    elif bf == 'parse_result':
        x_train_tfidf, x_test_tfidf, x_train_pos_vec, x_test_pos_vec  = dependent_features_vectors([s.dependent_words for s in train_data],
                                                                 [s.dependent_words for s in test_data],
                                                                 [s.dependent_pos_tags for s in train_data],
                                                                 [s.dependent_pos_tags for s in test_data])
    elif bf == 'parse+chi':
        # x_train_tfidf, x_test_tfidf, _, _= bow_features_vectors([s.bow_words for s in train_data],
                                                           # [s.bow_words for s in test_data])
        x_train_tfidf, x_test_tfidf, x_train_pos_vec, x_test_pos_vec  = dependent_features_vectors([s.bow_words for s in train_data],
                                                     [s.bow_words for s in test_data],
                                                     [s.bow_tags for s in train_data],
                                                     [s.bow_tags for s in test_data])

    if lsa_k is not None and lsa_k != 'no':
        svd = TruncatedSVD(lsa_k, algorithm='arpack', random_state=42, n_iter=5000)
        lsa = make_pipeline(svd)
        x_train_tfidf = lsa.fit_transform(x_train_tfidf)
        x_test_tfidf = lsa.transform(x_test_tfidf)

    x_train_sbow = np.asarray([s.sbow_vec for s in train_data])
    x_test_sbow = np.asarray([s.sbow_vec for s in test_data])

    x_train_lfe = lexicons_features_vectors([s.words for s in train_data],
                                            [s.pos_tags for s in train_data],
                                            [s.dependent_words for s in train_data])
    x_test_lfe = lexicons_features_vectors([s.words for s in test_data],
                                           [s.pos_tags for s in test_data],
                                           [s.dependent_words for s in test_data])

    if asp:
        x_train_asp_emb = np.asarray([get_aspect_embedding(s) for s in train_data])
        x_test_asp_emb =  np.asarray([get_aspect_embedding(s) for s in test_data])
        x_train = np.concatenate((x_train_tfidf, x_train_pos_vec,  x_train_sbow, x_train_lfe, x_train_asp_emb), axis=1)
        x_test = np.concatenate((x_test_tfidf, x_test_pos_vec, x_test_sbow, x_test_lfe, x_test_asp_emb), axis=1)
    else:
        x_train = np.concatenate((x_train_tfidf, x_train_pos_vec,  x_train_sbow, x_train_lfe), axis=1)
        x_test = np.concatenate((x_test_tfidf, x_test_pos_vec, x_test_sbow, x_test_lfe), axis=1)

    y_train = [y.polarity for y in train_data]
    y_test = [y.polarity for y in test_data]
    return x_train, y_train, x_test, y_test


def dependent_features_vectors(train_words, test_words, train_pos_tags=None, test_pos_tags=None):
    new_train_texts = []
    new_test_texts = []

    for words in train_words:
        new_words = [w for w in words if w not in stop_words]
        new_train_texts.append(" ".join(new_words))
    for words in test_words:
        new_words = [w for w in words if w not in stop_words]
        new_test_texts.append(" ".join(new_words))
    tfidf_vectorize = TfidfVectorizer(token_pattern=r'\w{1,}')
    x_train_tfidf = tfidf_vectorize.fit_transform(new_train_texts).toarray()
    x_test_tfidf = tfidf_vectorize.transform(new_test_texts).toarray()

    # add pos tags information
    x_train_pos_vec = []
    x_test_pos_vec = []
    if train_pos_tags is not None and test_pos_tags is not None:
        count_vectorize = CountVectorizer(token_pattern=r'\w{1,}', binary=False)
        new_train_pos = [" ".join(x) for x in train_pos_tags]
        new_test_pos = [" ".join(x) for x in test_pos_tags]
        x_train_pos_vec = count_vectorize.fit_transform(new_train_pos).toarray()
        x_test_pos_vec = count_vectorize.transform(new_test_pos).toarray()

    return x_train_tfidf, x_test_tfidf, x_train_pos_vec, x_test_pos_vec


def bow_features_vectors(train_sentences, test_sentences):
    tfidf_vectorize = TfidfVectorizer(token_pattern=r'\w{1,}')
    x_train_tfidf = tfidf_vectorize.fit_transform(train_sentences).toarray()
    x_test_tfidf = tfidf_vectorize.transform(test_sentences).toarray()

    return x_train_tfidf, x_test_tfidf


def lexicons_features_vectors(tokens, pos_tags, dependent_words=None):
    new_tokens = []
    new_pos_tags = []
    for words, tags, dw in zip(tokens, pos_tags, dependent_words):
        new_words = []
        new_tags = []
        # tmp_dw_set = set([w for w in dw if w not in stop_words])
        for w, t in zip(words, tags):
            # if w in tmp_dw_set:
            new_words.append(w)
            new_tags.append(t)
        new_tokens.append(new_words)
        new_pos_tags.append(new_tags)
    return LexiconFeatureExtractor(new_tokens, new_pos_tags).vectors


def evaluation(y_preds, y_true):
    acc = accuracy_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds, average='macro')
    clf_report = classification_report(y_true, y_preds)
    print("\n\n################################################################")
    print('Optimized acc: %.5f ' % acc)
    print('Optimized macro_f1: %.5f ' % f1)
    print(clf_report)
    print("####################################################################")

def load_tokenizer():
    global TOKENIZER
    # TODO: never split
    TOKENIZER = BertTokenizerFast.from_pretrained('bert-base-uncased')

def load_embed_dict(mode):
    global EMBED_DICT
    # mode 0 = wordpiece, mode 1 = neversplit
    if mode == 0:
        embed_path = os.path.join(REST_DIR, f'parsed_data/bert_embeddings_rest.plk')
    if mode == 1:
        embed_path = os.path.join(REST_DIR, f'parsed_data/bert_embeddings_rest_ns.plk')
    EMBED_DICT = pickle.load(open(embed_path, 'rb'))

def write_best_results(ht, r, aspect_id, cr, bf, iss, asp, incorrect_samples, suffix):
    if suffix != '':
        suffix = '_' + suffix
    Path(f"{REST_DIR}optimal_results/r{r}{suffix}/").mkdir(parents=True, exist_ok=True)
    with open(f"{REST_DIR}optimal_results/r{r}/svm_{str(aspect_id)}", 'w') as f:
        f.write("################################################################\n")
        f.write('chi_ratio: ' + str(cr) + '\n')
        # f.write('cr: ' + str(cr) + '\n')
        f.write('bow_features: ' + bf + '\n')
        f.write('is_sampling: ' + str(iss) + '\n')
        f.write('is_aspect_embeddings: ' + str(asp) + '\n')
        f.write(str(ht.best_cfg) + "\n")
        f.write('Optimized acc: %.5f \n' % ht.best_acc)
        f.write('Optimized macro_f1: %.5f \n' % ht.best_f1)
        f.write('training set shape: %s\n' % str(ht.train_X.shape))
        f.write(ht.clf_report)
        f.write("correct / total: %d / %d\n" % (ht.correct, len(ht.test_y)))
        f.write("elapsed time: %.5f s\n" % ht.elapsed_time)
        f.write(f'incorrect_samples:\n[{incorrect_samples}]')

def main():
    load_embed_dict(mode=0)
    load_tokenizer()
    chi_ratios = [x/10 for x in range(1, 11)]
    bow_features = ['all_words', 'parse_result', 'parse+chi']  #,'all_words',  'parse+chi'
    is_sampling = [True]
    is_aspect_embeddings = [True, False]
    best_accs = [0 for _ in range(0, 26)]
    print(chi_ratios)
    #num_rounds = 2000 # NOTE: CHANGE NO. ROUNDS
    num_rounds = 1
    suffix = ''
    for aspect_id in range(0, 20): # NOTE: CHANGE THIS RANGE TO REFLECT NUMBER OF CLUSTERS
        ht = HyperoptTunerLibSVM()
        # ht1 = HyperoptTunerLibSVM()
        for bf in bow_features:
            for iss in is_sampling:
                for asp in is_aspect_embeddings:
                    if 'chi' in bf:
                        for cr in chi_ratios:
                            data = Dataset(base_dir=REST_DIR, is_preprocessed=True, ratio=cr) #
                            train_data, test_data = data.data_from_aspect(aspect_id, is_sampling=iss)
                            print("aspect_cluster_id: %d, #train_instance = %d, #test_instance = %d" %
                                (aspect_id, len(train_data), len(test_data)))
                            x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bf, asp)
                            print(x_train.shape)
                            print(x_train)
                            scaler = Normalizer().fit(x_train)
                            x_train = scaler.transform(x_train)
                            x_test = scaler.transform(x_test)
                            ht.train_X = x_train
                            ht.train_y = y_train
                            ht.test_X = x_test
                            ht.test_y = y_test
                            ht.cluster_id = aspect_id
                            ht.base_dir = data.base_dir
                            ht.tune_params(num_rounds)

                            if ht.best_acc > best_accs[aspect_id]:
                                best_accs[aspect_id] = ht.best_acc
                                predictions = ht.pred_results.tolist()
                                true_labels = y_test
                                mask = [False if x[0] == x[1] else True for x in zip(predictions, true_labels)]
                                incorrect_samples = ', '.join([str(s.id) for i, s in enumerate(test_data) if mask[i]])
                                write_best_results(ht, num_rounds, aspect_id, cr, bf, iss, asp, incorrect_samples, suffix)

                    else:
                        data = Dataset(base_dir=REST_DIR, is_preprocessed=True) #
                        train_data, test_data = data.data_from_aspect(aspect_id, is_sampling=iss)
                        print("aspect_cluster_id: %d, #train_instance = %d, #test_instance = %d" %
                            (aspect_id, len(train_data), len(test_data)))
                        x_train, y_train, x_test, y_test = generate_vectors(train_data, test_data, bf, asp)
                    

if __name__ == '__main__':
    start = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start))
