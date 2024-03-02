# data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch

# Data Preparation
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.idx = 2  # Start indexing from 2

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx["<UNK>"])  # Return <UNK> index if word is not found

    def __len__(self):
        return len(self.word2idx)

def build_vocab(texts):
    vocab = Vocabulary()
    for text in texts:
        for word in text.split():
            vocab.add_word(word)
    return vocab


class RelationDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = [[vocab(word) for word in text.split()] for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return torch.tensor(self.texts[item], dtype=torch.long), torch.tensor(self.labels[item], dtype=torch.long)

def generate_loader(texts, labels, vocab, batch_size=32, shuffle=True):
    dataset = RelationDataset(texts, labels, vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts, labels

# Load dataset
dataset_path = 'train_dataset_dataframe.csv'
dataset = pd.read_csv(dataset_path)

# Tokenization and Encoding
texts = dataset['normalized_sentence'].values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(dataset['relation_type'].values)

vocab = build_vocab(texts)
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
#print(X_train, "//", X_val, "//", y_train, "//", y_val)