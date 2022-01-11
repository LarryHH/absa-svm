import os
import re
import configparser

NUMBERS = re.compile(r'(\d+)')

def numerical_sort(value):
    parts = NUMBERS.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def get_clf_names(fp):
    config = configparser.ConfigParser(allow_no_value=True)	
    config.optionxform = str	
    config.read(fp)
    clfs = config['CLASSIFIERS']
    return [c for c in clfs]

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

    return result_dict


if __name__ == '__main__':
    
    #result_dict = cal_acc('../svm-result/svm-result25')
    #result_dict = cal_acc('datasets/rest/tmp_optimized_result')
    base_dir = 'datasets/rest'
    fn = 'r500_k30_BERT'

    clf_names = get_clf_names('config/config.ini')

    for clf in clf_names:
        print(f"CLASSIFER: {clf}")
        result_dict = cal_acc(f'{base_dir}/optimal_results/{fn}', clf)
        if result_dict:
            correct = sum(result_dict['correct'])
            total = sum(result_dict['total'])
            f_scores = result_dict['f1']
            print(f'f_scores: {f_scores}')
            f1 = 0
            for num_sample, chunk_f in zip(result_dict['total'], f_scores):
                f1 += num_sample / total * chunk_f
            print('correct / total: %d / %d' % (correct, total))
            print('Acc: %.5f' % (correct / total))
            print('F1: %.5f' % f1)
