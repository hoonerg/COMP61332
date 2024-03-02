import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import KeyedVectors
import numpy as np

# Load the pre-trained Word2Vec model using Gensim's downloader
word2vec_path = 'GoogleNews-vectors-negative300.bin.gz'
word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

class Vocabulary:
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.embedding_dim = word2vec.vector_size

    def get_vector(self, word):
        try:
            return self.word2vec[word]
        except KeyError:
            return np.zeros(self.embedding_dim)
        
    def __len__(self):
        return len(self.word2vec.key_to_index)

class RelationDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        vectorized_text = [self.vocab.get_vector(word) for word in self.texts[item].split()]
        vectorized_text = np.stack(vectorized_text)
        return torch.tensor(vectorized_text, dtype=torch.float), torch.tensor(self.labels[item], dtype=torch.long)


def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return texts, labels

def generate_loader(texts, labels, vocab, batch_size=32, shuffle=True):
    dataset = RelationDataset(texts, labels, vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader

# Load dataset
dataset_path = 'train_dataset_dataframe.csv'
dataset = pd.read_csv(dataset_path)

# Tokenization and Encoding
texts = dataset['normalized_sentence'].values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(dataset['relation_type'].values)

# Initialize Vocabulary with the pre-loaded Word2Vec model
vocab = Vocabulary(word2vec)

# Splitting dataset
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Generate data loaders
#train_loader = generate_loader(X_train, y_train, vocab)
#val_loader = generate_loader(X_val, y_val, vocab, shuffle=False)
