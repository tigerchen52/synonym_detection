# Licensed under the GNU LGPL v2.1 - 


"""
USAGE: %(program)s -train CORPUS -output VECTORS -size SIZE -window WINDOW
-cbow CBOW -sample SAMPLE -hs HS -negative NEGATIVE -threads THREADS -iter ITER
-min_count MIN-COUNT -alpha ALPHA -binary BINARY -accuracy FILE


Parameters for training:
        -train <file>
                Use text data from <file> to train the model
        -output <file>
                Use <file> to save the resulting word vectors / word clusters
        -size <int>
                Set size of word vectors; default is 100
        -window <int>
                Set max skip length between words; default is 5
        -sample <float>
                Set threshold for occurrence of words. Those that appear with higher frequency in the training data
                will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
        -hs <int>
                Use Hierarchical Softmax; default is 0 (not used)
        -negative <int>
                Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
        -threads <int>
                Use <int> threads (default 3)
        -iter <int>
                Run more training iterations (default 5)
        -min_count <int>
                This will discard words that appear less than <int> times; default is 5
        -alpha <float>
                Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
        -binary <int>
                Save the resulting vectors in binary moded; default is 0 (off)
        -cbow <int>
                Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)
        -accuracy <file>
                Compute accuracy of the resulting model analogical inference power on questions file <file>

Example: python -m chi_annotator.algo_factory.preprocess.char2vec_standalone -train data.txt -output vec.txt -size 200 -sample 1e-4 -binary 0 -iter 3
"""

import numpy as np
import argparse
from numpy import seterr
seterr(all='raise')  # don't ignore numpy errors
from gensim.models.word2vec import Word2Vec, LineSentence  # avoid referencing __main__ in pickle
import logging
import os
import sys
logger = logging.getLogger(__name__)

local_file = os.path.split(__file__)[-1]
logging.basicConfig(
    format='%(asctime)s : %(filename)s : %(funcName)s : %(levelname)s : %(message)s',
    level=logging.INFO)
program = os.path.basename(local_file)

parser = argparse.ArgumentParser()
parser.add_argument("-train", help="Use text data from file TRAIN to train the model", default="../temp/segment_corpus.txt")
parser.add_argument("-output", help="Use file OUTPUT to save the resulting word vectors", default="../temp/w2v_embed_300.bin")
parser.add_argument("-window", help="Set max skip length WINDOW between words; default is 5", type=int, default=5)
parser.add_argument("-size", help="Set size of word vectors; default is 100", type=int, default=300)
parser.add_argument("-sample", help="Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled; "
                                    "default is 1e-3, useful range is (0, 1e-5)", type=float, default=1e-3)
parser.add_argument("-hs", help="Use Hierarchical Softmax; default is 0 (not used)", type=int, default=0, choices=[0, 1])
parser.add_argument("-negative", help="Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)", type=int, default=10)
parser.add_argument("-threads", help="Use THREADS threads (default 3)", type=int, default=3)
parser.add_argument("-iter", help="Run more training iterations (default 5)", type=int, default=10)
parser.add_argument("-min_count", help="This will discard words that appear less than MIN_COUNT times; default is 5", type=int, default=5)
parser.add_argument("-alpha", help="Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW", type=float)
parser.add_argument("-cbow", help="Use the continuous bag of words model; default is 1 (use 0 for skip-gram model)", type=int, default=0, choices=[0, 1])
parser.add_argument("-binary", help="Save the resulting vectors in binary mode; default is 0 (off)", type=int, default=0, choices=[0, 1])
parser.add_argument("-accuracy", help="Use questions from file ACCURACY to evaluate the model")
args = parser.parse_args()

def synonym_detect(input_word_code_dict, top_k):
    logger.info('start train w2v model.....')
    word2vec()
    logger.info('start w2v synonym detect......')
    cal_sim(args.output, input_word_code_dict, top_k)
    logger.info('w2v done!!!')



def word2vec():

    if args.cbow == 0:
        skipgram = 1
        if not args.alpha:
            args.alpha = 0.025
    else:
        skipgram = 0
        if not args.alpha:
            args.alpha = 0.05

    corpus = LineSentence(args.train)

    model = Word2Vec(
        corpus, size=args.size, min_count=args.min_count, workers=args.threads,
        window=args.window, sample=args.sample, alpha=args.alpha, sg=skipgram,
        hs=args.hs, negative=args.negative, cbow_mean=1, iter=args.iter)

    if args.output:
        outfile = args.output
        model.wv.save_word2vec_format(outfile, binary=args.binary)
    else:
        outfile = args.train.split('.')[0]
        model.save(outfile + '.model')
        if args.binary == 1:
            model.wv.save_word2vec_format(outfile + '.model.bin', binary=True)
        else:
            model.wv.save_word2vec_format(outfile + '.model.txt', binary=False)

    if args.accuracy:
        questions_file = args.accuracy
        model.accuracy(questions_file)

    logger.info("finished running %s", program)


def load_embedding(embed_path, input_word_code_dict):
    word_list = []
    word_embed = []
    input_word_ids = dict()
    with open(embed_path, encoding='utf8') as f:
        line = f.readline()
        line = f.readline()
        index = 0
        while line:
            row = line.strip().split(' ')
            word_list.append(row[0])
            if row[0] in input_word_code_dict:
                input_word_ids[index] = row[0]
            embed = [float(e) for e in row[1:]]
            word_embed.append(embed)
            line = f.readline()
            index += 1
    return word_list, np.array(word_embed), input_word_ids


def cal_sim(path, input_word_code_dict, top_k):
    word_list, word_embed, input_word_ids = load_embedding(path, input_word_code_dict)
    l2_word_embed = np.sqrt(np.sum(np.square(word_embed), axis=1))
    normal_word_embed = np.array([word_embed[i] / l2_word_embed[i] for i in range(len(word_embed))])
    input_word_embed = []
    input_word_list = []
    for index, word in input_word_ids.items():
        temp_embed = normal_word_embed[index]
        input_word_embed.append(temp_embed)
        input_word_list.append(word)
    input_word_embed = np.array(input_word_embed)
    normal_word_embed_T = normal_word_embed.T
    cos = np.matmul(input_word_embed, normal_word_embed_T)
    sorted_id = (-cos).argsort()
    line = ''
    for i, word in enumerate(input_word_list):
        code = input_word_code_dict[word]
        near_id = sorted_id[i][:top_k]
        nearst_word = [word_list[x] for x in near_id]
        line += code + '\t' + word + '\t' + '|'.join(nearst_word) + '\n'
    with open('../output/w2v_synonym.txt', 'w', encoding='utf8') as f:
        f.write(line)

def cal_sim_valid(path):
    word_list, word_embed, _ = load_embedding(path, {})
    l2_word_embed = np.sqrt(np.sum(np.square(word_embed), axis=1))
    normal_word_embed = np.array([word_embed[i] / l2_word_embed[i] for i in range(len(word_embed))])
    normal_word_embed_T = normal_word_embed.T
    cos = np.matmul(normal_word_embed, normal_word_embed_T)
    sorted_id = (-cos).argsort()
    line = ''
    for i in range(len(sorted_id)):
        near_id = sorted_id[i][:20]
        nearst_word = [word_list[x] for x in near_id]
        line += ','.join(nearst_word) + '\n'
    with open('../temp/embed_valid.txt', 'w', encoding='utf8') as f:
        f.write(line)

if __name__ == "__main__":
    cal_sim_valid(path='../temp/w2v_embed_300.bin')


