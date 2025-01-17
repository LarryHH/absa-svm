import stanza
import os
from file_utils import *
import pickle
from cluster_utils import *
from chi import CHI
from typing import List, Tuple

NLP_HELPER = None

def load_stanza():
    stanza.download('en')
    return stanza.Pipeline(lang='en', tokenize_pretokenized=True)
    
def aspect_cluster(dataset, ns=False, bert=True, n_clusters=20):
    if bert:
        ac = BERTAspectCluster(dataset, ns=ns, n_clusters=n_clusters)
    else:
        ac = AspectCluster(dataset, n_clusters=n_clusters)
    _, vectors = ac.fit()
    ac.predict()
    ac.save_cluster_result()

    return ac, vectors

def word_cluster(dataset, ns=False, bert=True, n_clusters=20):
    if bert:
        wc = BERTWordsCluster(dataset, ns=ns, n_clusters=n_clusters)
    else:
        wc = WordsCluster(dataset, n_clusters=n_clusters)
    wc.generate_vector()
    return wc

def chi_calculation(dataset, ratio):
    print(os.getcwd())
    stopwords = stop_words()
    chi_cal = CHI([" ".join(s.words) for s in dataset.train_data],
              [s.aspect_cluster for s in dataset.train_data],
              stopwords)

    chi_dict = {}
    for aspect_cluster, feature_list in chi_cal.chi_dict.items():
        chi_dict[aspect_cluster] = feature_list[0: int(len(feature_list) * ratio)]

    for sample in dataset.train_data:
        sample.bow_words = []
        sample.bow_tags = []
        for w in sample.words:
            if w in stopwords:
                continue
            if w in chi_dict[sample.aspect_cluster] or w in sample.dependent_words:
                sample.bow_words.append(w)
                sample.bow_tags.append(w)

    for sample in dataset.test_data:
        sample.bow_words = []
        sample.bow_tags = []
        for w in sample.words:
            if w in stopwords:
                continue
            if w in chi_dict[sample.aspect_cluster] or w in sample.dependent_words:
                sample.bow_words.append(w)
                sample.bow_tags.append(w)

class Dataset(object):
    def __init__(self, base_dir, is_preprocessed, ns=False, bert=True, aspect_clusters=20, word_clusters=20, ratio=0.3):
        self.base_dir = base_dir
        self.train_data = None
        self.test_data = None
        if not is_preprocessed:
            global NLP_HELPER
            NLP_HELPER = load_stanza()
            training_path = os.path.join(base_dir, 'train.txt')
            test_path = os.path.join(base_dir, 'test.txt')
            self.train_data = self.load_raw_data(training_path)
            self.test_data = self.load_raw_data(test_path)

            #print(self.train_data[0].__str__())
            self.preprocessing(self.train_data)
            self.preprocessing(self.test_data)


            print('attempt aspect cluster')
            aspect_cluster(self, ns, bert, aspect_clusters)
            print('attempt word cluster')
            word_cluster(self, ns, bert, word_clusters)

            print('save files...')
            self.save_as_pickle(base_dir, 'parsed_data', 'parsed_train.plk', 'parsed_test.plk', self.train_data, self.test_data)
            self.save_as_txt(base_dir, 'parsed_data', 'parsed_train.txt', 'parsed_test.txt', self.train_data, self.test_data)
        else:
            training_path = os.path.join(base_dir, 'parsed_data', 'parsed_train.plk')
            test_path = os.path.join(base_dir, 'parsed_data', 'parsed_test.plk')
            self.load_preprocessed_data(training_path, test_path)
            chi_calculation(self, ratio)

    def load_stanza(self):
        stanza.download('en')
        return stanza.Pipeline(lang='en', tokenize_pretokenized=True)

    @staticmethod
    def load_raw_data(path):
        data = []
        lines = read_as_list(path)
        for i in range(len(lines) // 3):
            text = lines[i * 3]
            aspect = lines[i * 3 + 1]
            polarity = int(lines[i * 3 + 2])       
            data.append(Sample(text, aspect, polarity, i))
        return data

    def load_preprocessed_data(self, training_path, test_path): 
        self.train_data = pickle.load(open(training_path, 'rb'))
        self.test_data = pickle.load(open(test_path, 'rb'))

    def format_hashstring(self, text: str, aspect: str) -> Tuple[List[str], str]:
        # fix malformed hashwords, e.g. '*##', '##*', '*##*', and if aspect has multiple terms, replace ## with aspects
        tokens = text.split(' ')
        idx = [i for i, token in enumerate(tokens) if '##' in token][0]
        hashwords = tokens[idx]
        hashwords = [w if w else "'" for w in hashwords.split("'")]
        tokens[idx] = '##'
        if len(hashwords) > 1:
            tokens = [y for x in tokens for y in ([x] if x != '##' else hashwords)]
        tokens = [t if '##' not in t else '##' for t in tokens]
        hashwords = [w for w in hashwords if w != "'"][0]
        # idx = tokens.index('##')
        aspect_trunc = ''
        # fix truncated aspect
        if len(hashwords) > 2:
            if hashwords.startswith('#'): # "##*"
                aspect_trunc = hashwords.split('##')[1]
                aspect = aspect + aspect_trunc
            elif hashwords.endswith('#'): # "*##"
                aspect_trunc = hashwords.split('##')[0] 
                aspect = aspect_trunc + aspect
            else: # "*##*"
                aspect_trunc = hashwords.split('##') 
                aspect = aspect.join(aspect_trunc)
        # replace '##' with aspects (single or multi)
        aspects = aspect.split(' ')
        tokens = [y for x in tokens for y in ([x] if x != '##' else aspects)]
        # if len(aspects) > 1:
        #     idx = idx + len(aspects) - 1
        return tokens, aspect
        
    def preprocessing(self, data):
        length = len(data)
        for i, sample in enumerate(data):
            # print(i*3)
            # print(sample.text)
            # print(sample.aspect)
            tokens, sample.aspect = self.format_hashstring(sample.text, sample.aspect)
            text = ' '.join(tokens)
            # print(tokens)
            # print(sample.aspect)
            nlp_parsed_obj = NLP_HELPER(text)
            sample.words, sample.pos_tags = list(map(list, zip(
                *[(word.text, word.xpos) for sent in nlp_parsed_obj.sentences for word in sent.words])))
            # print(sample.words)
            idx = sample.words.index(sample.aspect.split(' ')[0])
            #tmp_text = str.replace(sample.text, '##', sample.aspect)
            dependencies = [(dep_edge[1], dep_edge[0].id, dep_edge[2].id)
                            for sent in nlp_parsed_obj.sentences for dep_edge in sent.dependencies]
            sample.dependent_words, sample.dependent_pos_tags, _ = self.get_dependent_words(
                idx, dependencies, sample.words, sample.pos_tags, text, n=3, window_size=5)
            print(f'progress: {round(((i+1) / length * 100), 2)}% --- {i*3}')


    def data_from_aspect(self, aspect_cluster, is_sampling=True):
        pos = 0
        neg = 0
        net = 0
        train_samples = []
        for s in self.train_data:
            if s.aspect_cluster == aspect_cluster:
                if s.polarity == 1:
                    pos += 1
                elif s.polarity == 0:
                    net += 1
                else:
                    neg += 1
                train_samples.append(s)
        if is_sampling:
            if net < pos:
                for s in self.train_data:
                    if s.polarity == 0 and s.aspect_cluster != aspect_cluster:
                        train_samples.append(s)
                        net += 1
                    if net >= pos:
                        break
            if neg < pos:
                for s in self.train_data:
                    if s.polarity == -1 and s.aspect_cluster != aspect_cluster:
                        train_samples.append(s)
                        neg += 1
                    if neg >= pos:
                        break
        test_samples = [s for s in self.test_data if s.aspect_cluster == aspect_cluster]

        return train_samples, test_samples

    def get_aspect_labels(self):
        return list(set([s.aspect_cluster for s in self.train_data]))

    def direction_dependent(self, temp_dict, word, n):
        selected_words = []
        if word not in temp_dict.keys():
            return []
        else:
            tmp_list = temp_dict[word]
            selected_words.extend(tmp_list)
            if n > 1:
                for w in tmp_list:
                    selected_words.extend(self.direction_dependent(temp_dict, w, n - 1))
        return selected_words

    def get_dependent_words(self, idx, dependent_results, words, pos_tags, text, n=2, window_size=0):
        
        #dependent_results = dependent_parse(text)
        in_dict = {}
        out_dict = {}
        for dr in dependent_results:
            # print(dr[0])
            src_wid = dr[1]    # source wid
            tag_wid = dr[2]    # target wid
            out_dict.setdefault(src_wid, [])
            in_dict.setdefault(tag_wid, [])

            out_dict[src_wid].append(tag_wid)
            in_dict[tag_wid].append(src_wid)

        forwards = self.direction_dependent(out_dict, idx + 1, n)
        backwards = self.direction_dependent(in_dict, idx + 1, n)

        result = []
        result.extend(forwards)
        result.extend(backwards)

        # add window-size words
        if window_size != 0:
            # right side
            for i in range(idx + 2, idx + 2 + window_size, 1):
                if i > len(words):
                    break
                result.append(i)
            for i in range(idx + 1 - window_size, idx + 1, 1):
                if i > 1:
                    result.append(i)
        result = list(set(result))
        result.sort()

        #print("!!!!!!!--->> " + " ".join([pos_tags[i-1] for i in result]))

        return [words[i-1] for i in result], [pos_tags[i-1] for i in result], dependent_results


    def save_as_pickle(self, base_dir, fp, fname_tr, fname_t, train_data, test_data):
        training_path = os.path.join(base_dir, fp, fname_tr)
        test_path = os.path.join(base_dir, fp, fname_t)
        pickle.dump(train_data, open(training_path, 'wb'))
        pickle.dump(test_data, open(test_path, 'wb'))

    def save_as_txt(self, base_dir, fp, fname_tr, fname_t, train_data, test_data):
        training_path = os.path.join(base_dir, fp, fname_tr)
        test_path = os.path.join(base_dir, fp, fname_t)
        with open(training_path, 'w') as f:
            for sample in train_data:
                f.write(sample.__str__())
        with open(test_path, 'w') as f:
            for sample in test_data:
                f.write(sample.__str__())

    def save_as_tmp(self, base_dir, fp_tr, fp_t, train_data, test_data):
        remove_dirs(base_dir)
        make_dirs(os.path.join(base_dir, 'train'))
        make_dirs(os.path.join(base_dir, 'test'))
        for s in train_data:
            with open(f"{base_dir}/{fp_tr}/" + str(s.aspect_cluster), 'a') as f:
                f.write(s.text + "\n")
                f.write(s.aspect + "\n")
                f.write(str(s.polarity) + "\n")
        for s in test_data:
            with open(f"{base_dir}/{fp_t}/" + str(s.aspect_cluster), 'a') as f:
                f.write(s.text + "\n")
                f.write(s.aspect + "\n")
                f.write(str(s.polarity) + "\n")


class Sample(object):
    def __init__(self, text, aspect, polarity, sample_id=0):
        self.id = sample_id
        self.text = text
        self.aspect = aspect
        self.polarity = polarity
        self.words = []
        self.pos_tags = []
        self.dependent_words = []   # words that has dependency with aspect
        self.dependent_pos_tags = []
        self.aspect_cluster = -1
        self.bow_words = []
        self.bow_tags = []
        self.sbow_vec = []

    def __str__(self):
        result = "###############################################################\n" + \
                 self.text + '\n' + self.aspect + '\n' + str(self.polarity) + '\n' + \
                 str(self.aspect_cluster) + '\n' + " ".join(self.words) + '\n' + " ".join(self.pos_tags)\
                 + '\n' + " ".join(self.dependent_words) + '\n' + " ".join(self.dependent_pos_tags) + '\n'\
                 "###############################################################\n"

        return result



# if __name__ == '__main__':
#     base_dir = 'datasets/rest/'
#     data = Dataset(base_dir, is_preprocessed=False)


if __name__ == "__main__":
    base_dir = 'datasets/rest/'
    data = Dataset(base_dir, is_preprocessed=False)