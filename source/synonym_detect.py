import Levenshtein_model
import data_utils
import semantic_network_model
import baike_crawler_model
import word2vec_model
import argparse
import logging
import os
import sys

parser = argparse.ArgumentParser(description='synonym_detect')
parser.add_argument('-corpus_path', help='use this text to train model', type=str, default='../input/三体.txt')
parser.add_argument('-input_word_path', help='find synonyms of these word, two columns, use | as segment ', type=str, default='../temp/input_word.txt')
parser.add_argument('-stop_word_path', type=str, default='../input/stop_words.txt')
parser.add_argument('-process_number', help='set the number of process, default is 30', type=int, default=30)
parser.add_argument('-if_use_pinyin', help='if use the pinyin to calculate similarity between two words, default is false', type=bool, default=False)
parser.add_argument('-pinyin_weight', help='set the weight of similarity of pinyin, default is 0.0', type=float, default=0.0)
parser.add_argument('-top_k', help='set the number of synonym we want to get, default is 5', type=int, default=5)
parser.add_argument('-win_len', help='set the window size of semantic network, default is 5', type=int, default=5)
parser.add_argument('-if_use_sn_model', help='use semantic model , default is False', type=bool, default=False)
parser.add_argument('-if_use_leven_model', help='use Levenshtein model , default is False', type=bool, default=False)
parser.add_argument('-if_use_baike_crawler', help='use baike crawler, this model need to connect to the network, default is False', type=bool, default=False)
parser.add_argument('-if_use_w2v_model', help='use word2vector model , default is False', type=bool, default=False)
args = parser.parse_args()

logger = logging.getLogger(__name__)
local_file = os.path.split(__file__)[-1]
logging.basicConfig(
    format='%(asctime)s : %(filename)s : %(funcName)s : %(levelname)s : %(message)s',
    level=logging.INFO)



def run():

    stop_word_path = args.stop_word_path
    corpus_path = args.corpus_path
    input_word_path = args.input_word_path
    process_number = args.process_number
    if_use_pinyin = args.if_use_pinyin
    pinyin_weight = args.pinyin_weight
    top_k = args.top_k
    win_len = args.win_len
    if_use_sn_model = args.if_use_sn_model
    if_use_leven_model = args.if_use_leven_model
    if_use_baike_crawler = args.if_use_baike_crawler
    if_use_w2v_model = args.if_use_w2v_model


    word2id, word_list, id2word, input_word_code_dict, input_word_id = \
        data_utils.preprocess_file(corpus_path, input_word_path, stop_word_path)

    if if_use_sn_model:
        semantic_network_model.synonym_detect(
            corpus_path=corpus_path,
            input_word_id=input_word_id,
            input_word_code_dict=input_word_code_dict,
            id2word=id2word,
            word2id=word2id,
            top_k=top_k,
            win_len=win_len,
            process_number=process_number
        )

    if if_use_leven_model:
        l_model = Levenshtein_model.Levenshtein_model(
            input_word=list(input_word_code_dict.keys()),
            candidate_word=word_list,
            process_number=process_number,
            if_use_pinyin=if_use_pinyin,
            pinyin_weight=pinyin_weight,
            top_k=top_k
        )
        l_model.multipro_synonym_detect(input_word_code_dict)

    if if_use_baike_crawler:
        word_code_list = list()
        for k, v in input_word_code_dict.items():
            word_code_list.append((k, v))
        baike_crawler_model.baike_synonym_detect(word_code_list)

    if if_use_w2v_model:
        word2vec_model.synonym_detect(input_word_code_dict, top_k)


if __name__ == '__main__':
    logger.info("running %s", " ".join(sys.argv))
    run()