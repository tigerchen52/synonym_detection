import Levenshtein
import pinyin
from multiprocessing import Process, Lock
import os
import logging

logger = logging.getLogger(__name__)
local_file = os.path.split(__file__)[-1]
logging.basicConfig(
    format='%(asctime)s : %(filename)s : %(funcName)s : %(levelname)s : %(message)s',
    level=logging.INFO)


class Levenshtein_model():

    def __init__(
            self,
            input_word=None,
            candidate_word=None,
            if_use_pinyin=False,
            pinyin_weight=0.2,
            top_k=15,
            process_number=10
    ):
        # 初始化，没什么特别的，就是把外部的参数传到对象里去
        self.base_word = input_word
        self.candidate_word = candidate_word
        self.if_use_pinyin = if_use_pinyin
        self.pinyin_weight = pinyin_weight
        self.top_k = top_k
        self.processe_number = process_number

        self.can_py_dict = {}

        if if_use_pinyin:
            for w in self.candidate_word:
                py = get_pinyin(w)
                self.can_py_dict[w] = py

    def calculate_similarity(self, w1, w2):
        # 这主要就是调用 Levenshtein 的 ratio 方法来获取两个词的相似度
        leven_dis = Levenshtein.ratio(w1, w2)
        leven_py_dis = 0.0
        if self.if_use_pinyin:
            py1 = get_pinyin(w1)
            py2 = self.can_py_dict[w2]
            leven_py_dis = Levenshtein.ratio(py1, py2)
        similarity = (1-self.pinyin_weight)*leven_dis + self.pinyin_weight*leven_py_dis
        return similarity

    def synonym_detect(self, base_word, lock, input_word_code_dict):
        # 核心处理过程
        word_synonym = {}
        cnt = 0
        rs = {}
        for word in base_word:
            cnt += 1
            logger.info('process {b}, calculating the {a} base word......'.format(a=cnt, b=os.getpid()))
            dis_dict = {}
            for candi in self.candidate_word:
                # 调用 calculate_similarity 方法计算两个词的相似度
                # 这个相似度是判断近义词的重要标准
                dis = self.calculate_similarity(word, candi)
                dis_dict[candi] = dis
            synonyms = sorted(dis_dict.items(), key=lambda e:e[1], reverse=True)
            word_synonym[word] = synonyms[0:self.top_k]

            # 把满足条件的近义词存入一个集合中
            # rs 结构为 {'原词': [('近义词1', '0.91'), ('近义词2', '0.68'), ...], ...}
            rs[word] = []
            for nword, score in word_synonym[word]:
                if word == nword:
                    continue
                rs[word].append((nword, score))

        # 这是把结果写入文件，我们现在用 mysql 存储，所以下面这段可以删掉不用
        line = ""
        for word in word_synonym.keys():
            values = word_synonym[word]
            synonyms = [s[0] for s in values]
            word_code = input_word_code_dict[word]
            synonyms = word_code + '\t' + word + '\t' + '|'.join(synonyms)
            line += synonyms + '\n'
        logger.info('process {a} starting writing to file ......'.format(a=os.getpid()))
        with lock:
            with open('../output/Levenshtein_model_synonym.txt', 'a', encoding='utf8') as f:
                f.write(line)

        return rs

    def multipro_synonym_detect(self, input_word_code_dict):
        if os.path.exists('../output/Levenshtein_model_synonym.txt'):
            os.remove('../output/Levenshtein_model_synonym.txt')
        lock = Lock()
        # 原来这里为了加快速度，使用多进程，并发异步处理
        # 因为我们现在要同步的输出到 mysql，所以这里改成单进程运行
        # 也就是直接调用 synonym_detect 方法
        rs = self.synonym_detect(self.base_word, lock, input_word_code_dict)
        # import math
        # logger.info('start detecting synonym ......')
        # partition = math.ceil(len(self.base_word) / self.processe_number)
        # start, end = 0, partition
        # pro_list = []
        # word_num = len(self.base_word)
        # if word_num < self.processe_number:
        #     logger.info('error, process number more than the amount of base word!')
        #     return
        # for i in range(self.processe_number):
        #     if end > word_num: break
        #     word_id = self.base_word[start:end]
        #     p = Process(target=self.synonym_detect, args=(word_id, lock, input_word_code_dict))
        #     pro_list.append(p)
        #     p.start()
        #     start, end = end, min(end + partition, word_num)
        # for p in pro_list:
        #     p.join()
        logger.info('finished！......')
        return rs


def get_pinyin(chinese_word):
    return pinyin.get(chinese_word, format="strip", delimiter=" ")


if __name__ == '__main__':
    dis = Levenshtein_model('../temp/base_word.txt','../temp/word2id')

