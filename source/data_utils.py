import jieba
import nltk
import logging
import os

logger = logging.getLogger(__name__)
local_file = os.path.split(__file__)[-1]
logging.basicConfig(
    format='%(asctime)s : %(filename)s : %(funcName)s : %(levelname)s : %(message)s',
    level=logging.INFO)
# jieba.load_userdict('../input/word.dict')


def get_stop_words(path):
    stop_words = set()
    with open(path, encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.strip()
            stop_words.add(line)
            line = f.readline()
    stop_words = set(stop_words)
    return stop_words


def segment(row=None, method='jieba'):
    if method == 'jieba':
        return jieba.cut(row, cut_all=False)
    else:
        return nltk.word_tokenize(row)

def remove_parentheses(entity):
    keys = {'［', '(', '[', '（'}
    symbol = {'］':'［', ')':'(', ']':'[', '）':'（'}
    stack = []
    remove = []
    for index, s in enumerate(entity):
        if s in keys:
            stack.append((s, index))
        if s in symbol:
            if not stack:continue
            temp_v, temp_index = stack.pop()
            if entity[index-1] == '\\':
                t = entity[temp_index-1:index+1]
                remove.append(t)
            else:
                remove.append(entity[temp_index:index+1])

    for r in remove:
        entity = entity.replace(r, '')
    return entity


def word_id_file(text_path='../temp/corpus.txt', stop_word_path='../input/stop_words.txt'):
    stop_words = get_stop_words(stop_word_path)

    entity_set = set()

    segment_path = '../temp/segment_corpus.txt'
    segment_file = open(segment_path, 'w',encoding='utf8',errors='ignore')

    with open(text_path, "r", encoding='utf-8', errors='ignore') as f:
        line = f.readline()
        while line:
            row = line.strip()
            raw_words = list(segment(row))
            segment_file.write(' '.join(raw_words) + '\n')
            for word in raw_words:
                if word not in stop_words:
                    entity_set.add(word)
            line = f.readline()

    line = '<PAD>,0\n'
    cnt = 1
    for e in entity_set:
        line += e + "," + str(cnt) + "\n"
        cnt += 1

    logger.info("file has created，totally {a} words .".format(a=len(entity_set)))
    with open("../temp/word2id", 'w', encoding='utf8') as f:
        f.write(line)


def load_word2id(path="../temp/word2id"):
    word2id = {}
    id2word = {}
    words = []
    with open(path, encoding='utf8') as f:
        line = f.readline()
        while line:
            row = line.strip().split(",")
            word2id[row[0]] = int(row[1])
            id2word[int(row[1])] = row[0]
            words.append(row[0])

            line = f.readline()
    logger.info('file has loaded，totally {a} words .'.format(a=len(words)))
    return word2id, words, id2word


def load_input_words(base_word_path, word2id, id2word):
    input_word_code_dict = dict()
    index = len(word2id)
    # with open(base_word_path, "r", encoding='utf-8', errors='ignore') as f:
    #     line = f.readline()
    #     while line:
    #         row = line.strip().split('|')
    #         word, word_code = row[1], row[0]
    #         if word2id is not None and word not in word2id:
    #             word2id[word] = index
    #             id2word[index] = word
    #             index += 1
    #         input_word_code_dict[word] = word_code
    #         line = f.readline()
    # logger.info('totally {a} words .'.format(a=len(input_word_code_dict)))

    content = open(base_word_path, "r", encoding='utf-8', errors='ignore').read().strip()
    words = content.split(',')
    i = 1
    for word in words:
        word_code = str(i)
        if word2id is not None and word not in word2id:
            word2id[word] = index
            id2word[index] = word
            index += 1
        input_word_code_dict[word] = word_code
        i += 1

    return input_word_code_dict


def preprocess_file(corpus_path, input_word_path, stop_word_path):
    logger.info('start preprocess......')
    word_id_file(corpus_path, stop_word_path)

    word2id, word_list, id2word = load_word2id()

    input_word_code_dict = load_input_words(input_word_path, word2id, id2word)

    input_word_id = [word2id[w] for w in list(input_word_code_dict.keys())]

    logger.info(' done!!!')
    return word2id, word_list, id2word, input_word_code_dict, input_word_id



