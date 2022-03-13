from registry import register
from functools import partial
import torch
import random
from torch.utils.data import DataLoader
from utils import load_dataset, TextData, TOKENIZER, repre_word, load_neg_samples
from attacks import get_random_attack
registry = {}
register = partial(register, registry=registry)


@register('simple')
class SimpleLoader():
    def __init__(self, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def collate_fn(self, batch_data, pad=0):
        batch_words, batch_oririn_repre = list(zip(*batch_data))

        aug_words, aug_repre, aug_ids = list(), list(), list()
        for index in range(len(batch_words)):
            aug_word = batch_words[index]
            repre, repre_ids = repre_word(aug_word, TOKENIZER, rtype='mixed')
            aug_words.append(aug_word)
            aug_repre.append(repre)
            aug_ids.append(repre_ids)

        batch_words = list(batch_words) + aug_words
        batch_oririn_repre = torch.FloatTensor(batch_oririn_repre)

        x_lens = [len(x) for x in aug_ids]
        max_len = max([len(seq) for seq in aug_ids])
        batch_aug_repre_ids = [char + [pad] * (max_len - len(char)) for char in aug_ids]
        batch_aug_repre_ids = torch.LongTensor(batch_aug_repre_ids)
        mask = torch.ne(batch_aug_repre_ids, pad).unsqueeze(2)
        return batch_words, batch_oririn_repre, batch_aug_repre_ids, mask

    def __call__(self, data_path, neg_sample_path=''):
        #neg_samples = load_neg_samples(path='data/hard_neg_samples.txt')
        dataset, _ = load_dataset(path=data_path)
        dataset = TextData(dataset)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size // 2, shuffle=self.shuffle,
                                    collate_fn=self.collate_fn)
        return train_iterator


@register('aug')
class SimpleLoader():
    def __init__(self, batch_size, probs, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.probs = probs

    def collate_fn(self, batch_data, pad=0):
        batch_words, batch_oririn_repre = list(zip(*batch_data))

        aug_words, aug_repre, aug_ids = list(), list(), list()
        for index in range(len(batch_words)):
            aug_word = get_random_attack(batch_words[index], self.probs)
            repre, repre_ids = repre_word(aug_word, TOKENIZER, rtype='mixed')
            aug_words.append(aug_word)
            aug_repre.append(repre)
            aug_ids.append(repre_ids)

        batch_words = list(batch_words) + aug_words
        batch_oririn_repre = torch.FloatTensor(batch_oririn_repre)

        x_lens = [len(x) for x in aug_ids]

        max_len = max([len(seq) for seq in aug_ids])
        batch_aug_repre_ids = [char + [pad] * (max_len - len(char)) for char in aug_ids]
        batch_aug_repre_ids = torch.LongTensor(batch_aug_repre_ids)

        mask = torch.ne(batch_aug_repre_ids, pad).unsqueeze(2)
        return batch_words, batch_oririn_repre, batch_aug_repre_ids, mask

    def __call__(self, data_path):
        dataset, _ = load_dataset(path=data_path)
        dataset = TextData(dataset)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size // 2, shuffle=self.shuffle,
                                    collate_fn=self.collate_fn)
        return train_iterator


@register('hard')
class SimpleLoader():
    def __init__(self, batch_size, neg_numbers=4, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.neg_numbers = neg_numbers-1
        self.neg_samples = load_neg_samples('data/hard_neg_samples.txt')
        self.all_words = list(self.neg_samples.keys())
        self.emb = None

    def collate_fn(self, batch_data, pad=0):
        batch_words, batch_oririn_repre = list(zip(*batch_data))

        batch_words_with_hards, batch_repre_with_hards = list(), list()
        for word in batch_words:
            if word in self.neg_samples:
                neg_words = self.neg_samples[word]
            else:
                neg_words = []
            if len(neg_words) >= self.neg_numbers:
                batch_hards = list(random.sample(neg_words, self.neg_numbers))
            else:
                sum_words = list(random.sample(self.all_words, self.neg_numbers - len(neg_words)))
                batch_hards = list(neg_words + sum_words)
            batch_words_with_hards.append(word)
            batch_words_with_hards.extend(batch_hards)
            batch_repre_with_hards.append(self.emb[word])
            for w in batch_hards:
                if w not in self.emb:
                    print('this word {a} does not in vocab'.format(a=w))
            batch_repre_with_hards.extend([self.emb[w] if w in self.emb else self.emb['the'] for w in batch_hards])

        aug_words, aug_repre, aug_ids = list(), list(), list()
        for index in range(len(batch_words_with_hards)):
            aug_word = get_random_attack(batch_words_with_hards[index])
            repre, repre_ids = repre_word(aug_word, TOKENIZER, id_mapping=None, rtype='mixed')
            aug_words.append(aug_word)
            aug_repre.append(repre)
            aug_ids.append(repre_ids)

        batch_words = batch_words_with_hards + aug_words
        batch_repre_with_hards = torch.FloatTensor(batch_repre_with_hards)

        x_lens = [len(x) for x in aug_ids]

        max_len = max([len(seq) for seq in aug_ids])
        batch_aug_repre_ids = [char + [pad] * (max_len - len(char)) for char in aug_ids]
        batch_aug_repre_ids = torch.LongTensor(batch_aug_repre_ids)

        mask = torch.ne(batch_aug_repre_ids, pad).unsqueeze(2)
        return batch_words, batch_repre_with_hards, batch_aug_repre_ids, mask

    def __call__(self, data_path):
        dataset, self.emb = load_dataset(path=data_path)
        dataset = TextData(dataset)
        train_iterator = DataLoader(dataset=dataset, batch_size=self.batch_size // (2 * (self.neg_numbers+1)), shuffle=self.shuffle, collate_fn=self.collate_fn)
        return train_iterator