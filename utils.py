import tokenization
import torch
import collections
from torch.utils.data import Dataset

VOCAB = 'data/vocab.txt'
TOKENIZER = tokenization.FullTokenizer(vocab_file=VOCAB, do_lower_case=True)
DIM = 300


def load_dataset(path):
    origin_words, origin_repre = list(), list()
    all_embs = dict()
    cnt = 0
    for line in open(path, encoding='utf8'):
        cnt += 1
        if cnt == 1:continue
        row = line.strip().split(' ')
        if len(row) != DIM + 1:continue
        word = str.lower(row[0])
        if filter(word): continue
        emb = [float(e) for e in row[1:]]
        origin_repre.append(emb)
        origin_words.append(word)
        all_embs[word] = emb

    print('loaded! Word num = {a}'.format(a=len(origin_words)))
    return {'origin_word': origin_words, 'origin_repre':origin_repre}, all_embs


def load_predict_dataset(path):
    origin_words, origin_repre = list(), list()
    for line in open(path, encoding='utf8'):
        word = line.strip()
        origin_repre.append(word)
        origin_words.append(word)
    print('loaded! Word num = {a}'.format(a=len(origin_words)))
    return {'origin_word': origin_words, 'origin_repre':origin_repre}


class TextData(Dataset):
    def __init__(self, data):
        self.origin_word = data['origin_word']
        self.origin_repre = data['origin_repre']
        #self.repre_ids = data['repre_ids']

    def __len__(self):
        return len(self.origin_word)

    def __getitem__(self, idx):
        return self.origin_word[idx], self.origin_repre[idx]


def collate_fn(batch_data, pad=0):
    batch_words, batch_oririn_repre = list(zip(*batch_data))

    aug_words, aug_repre, aug_ids = list(), list(), list()
    for index in range(len(batch_words)):
        #aug_word = get_random_attack(batch_words[index])
        aug_word = batch_words[index]
        repre, repre_ids = repre_word(aug_word, TOKENIZER, id_mapping=None)
        aug_words.append(aug_word)
        aug_repre.append(repre)
        aug_ids.append(repre_ids)

    batch_words = list(batch_words) + aug_words
    batch_oririn_repre = torch.FloatTensor(batch_oririn_repre)

    x_lens = [len(x) for x in aug_ids]
    max_len = max([len(seq) for seq in aug_ids])
    batch_aug_repre_ids = [char + [pad]*(max_len - len(char)) for char in aug_ids]
    batch_aug_repre_ids = torch.LongTensor(batch_aug_repre_ids)

    return batch_words, batch_oririn_repre, batch_aug_repre_ids, x_lens


def collate_fn_predict(batch_data, pad=0):
    batch_words, batch_oririn_repre = list(zip(*batch_data))

    batch_repre_ids = list()
    for word in batch_words:
        repre, repre_id = repre_word(word, TOKENIZER, id_mapping=None, rtype='mixed')
        batch_repre_ids.append(repre_id)

    x_lens = [len(x) for x in batch_repre_ids]
    max_len = max([len(seq) for seq in batch_repre_ids])
    batch_repre_ids = [char + [pad]*(max_len - len(char)) for char in batch_repre_ids]
    batch_repre_ids = torch.LongTensor(batch_repre_ids)
    mask = torch.ne(batch_repre_ids, pad).unsqueeze(2)
    return batch_words, batch_oririn_repre, batch_repre_ids, mask


def filter(word):
    min_len = 1
    if len(word) < min_len:return True
    return False


def tokenize_and_getid(word, tokenizer):
    tokens = tokenizer.tokenize(tokenizer.convert_to_unicode(word))
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, token_ids


def hash_sub_word(total, bucket):
    bucket -= 1
    id_mapping = collections.OrderedDict()
    for id in range(total):
        hashing = ((id % bucket) ^ 2) + 1
        #print(id, hashing)
        id_mapping[id] = hashing
    id_mapping[0] = 0
    id_mapping[100] = bucket + 2
    id_mapping[101] = bucket + 3
    id_mapping[102] = bucket + 4
    id_mapping[103] = bucket + 5
    id_mapping[104] = bucket + 6
    return id_mapping


def repre_word(word, tokenizer, id_mapping=None, rtype='mixed'):
    start = '[CLS]'
    sub = '[SUB]'
    end = '[SEP]'
    char_seq = list(word)
    tokens, _ = tokenize_and_getid(word, tokenizer)

    if rtype == 'mixed':
        repre = [start] + char_seq + [sub] + tokens + [end]
    elif rtype == 'char':
        repre = [start] + char_seq + [end]
    else:
        repre = [start] + tokens + [end]
    repre_ids = tokenizer.convert_tokens_to_ids(repre)
    if id_mapping:
        repre_ids = [id_mapping[r_id] for r_id in repre_ids]
    return repre, repre_ids


def load_neg_samples(path):
    neg_samples = dict()
    for line in open(path, encoding='utf8'):
        row = line.strip().split('\t')
        neg_samples[row[0]] = row[1:]
    return neg_samples


if __name__ == '__main__':
    pass


