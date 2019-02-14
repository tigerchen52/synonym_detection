"""
@version: ??
@author: lihu.clh
@file: main.py
@time: 2019/2/13 17:16
@desc:
"""
def load():
    w_l = ''
    for line in open('./temp/疾病术语集.txt', encoding='utf8'):
        row = line.strip().split('\t')
        if len(row) != 3: continue
        icd_code, word = row[1], row[2]
        w_l += icd_code + '|' + word + '\n'
    with open('./temp/input_disease_word.txt', 'w', encoding='utf8')as f:
        f.write(w_l)

def load_synonym(path, seg='|'):
    synonym_dict = dict()
    for line in open(path, encoding='utf8'):
        row = line.strip().split('\t')
        code, word, synonyms = row[0], row[1], row[2].split(seg)
        synonym_dict[code] = synonyms
    print('[ load_synonym ] path = {a}, word number = {b}'.format(a=path, b=len(synonym_dict)))
    return synonym_dict

def load_term():
    term_dict = dict()
    for line in open('./temp/疾病术语集.txt',encoding='utf8'):
        row = line.strip().split('\t')
        code, word = row[1], row[2]
        term_dict[code] = word
    print('[ load_term ] word number = {b}'.format(b=len(term_dict)))
    return term_dict


def merge_synonym():
    path1 = './temp/merge_synonym.tsv'
    path2 = './output/baike_synonym.txt'
    path3 = './output/Levenshtein_model_synonym.txt'
    path4 = './output/semantic_network_model_synonym.txt'
    s1 = load_synonym(path1)
    s2 = load_synonym(path2)
    s3 = load_synonym(path3)
    s4 = load_synonym(path4, seg='||')

    result = dict()
    term_dict = load_term()
    for code, word in term_dict.items():
        result[code] = list()

    def merge(result, synonym_dict):
        for c, synonyms in synonym_dict.items():
            if c not in result:continue
            exist = result[c]
            for s in synonyms:
                if s not in exist:
                    exist.append(s)

    merge(result, s1)
    merge(result, s2)
    merge(result, s3)
    merge(result, s4)

    w_l = ''
    for code, synonyms in result.items():
        word = term_dict[code]
        w_l += code + '\t' + word + '\t' + '|'.join(synonyms)+ '\n'

    with open('./output/synonyms.txt','w',encoding='utf8')as f:
        f.write(w_l)



if __name__ == '__main__':
    merge_synonym()