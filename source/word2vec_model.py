import numpy as np
import argparse
from numpy import seterr
seterr(all='raise')
from gensim.models.word2vec import Word2Vec, LineSentence
import logging
import os
logger = logging.getLogger(__name__)

local_file = os.path.split(__file__)[-1]
logging.basicConfig(
    format='%(asctime)s : %(filename)s : %(funcName)s : %(levelname)s : %(message)s',
    level=logging.INFO)
program = os.path.basename(local_file)

w2v_train = "../temp/segment_corpus.txt"
w2v_output = "../temp/w2v_embed_300.bin"
w2v_window = 5
w2v_size = 300
w2v_sample = 1e-3
w2v_hs = 0
w2v_negative = 10
w2v_threads = 3
w2v_iter = 10
w2v_min_count = 5
w2v_alpha = None
w2v_cbow = 0
w2v_binary = 0
w2v_accuracy = None

def synonym_detect(input_word_code_dict, top_k):
    logger.info('start train w2v model.....')
    word2vec()
    logger.info('start w2v synonym detect......')
    cal_sim(w2v_output, input_word_code_dict, top_k)
    logger.info('w2v done!!!')



def word2vec():

    if w2v_cbow == 0:
        skipgram = 1
        w2v_alpha = 0.025
    else:
        skipgram = 0
        w2v_alpha = 0.05

    corpus = LineSentence(w2v_train)

    model = Word2Vec(
        corpus, size=w2v_size, min_count=w2v_min_count, workers=w2v_threads,
        window=w2v_window, sample=w2v_sample, alpha=w2v_alpha, sg=skipgram,
        hs=w2v_hs, negative=w2v_negative, cbow_mean=1, iter=w2v_iter)

    if w2v_output:
        outfile = w2v_output
        model.wv.save_word2vec_format(outfile, binary=w2v_binary)
    else:
        outfile =w2v_train.split('.')[0]
        model.save(outfile + '.model')
        if w2v_binary == 1:
            model.wv.save_word2vec_format(outfile + '.model.bin', binary=True)
        else:
            model.wv.save_word2vec_format(outfile + '.model.txt', binary=False)

    if w2v_accuracy:
        questions_file = w2v_accuracy
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


