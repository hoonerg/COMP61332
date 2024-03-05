import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import KeyedVectors
import numpy as np

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


def find_max_word_length(sentences):
    max_lengths = []
    for sentence in sentences:
        # Splitting the sentence into words and finding the length of each word
        word_lengths = [len(word) for word in sentence.split()]
        # Finding the maximum length in the current sentence
        max_length = max(word_lengths)
        max_lengths.append(max_length)
    return max(max_lengths)

def vectorize_sentence(sentences, max_length= 30):
    vector_lists = []
    
    for sentence in sentences:
        vector = vectorize_words(sentence, max_length)
        vector_lists.append(vector)
    
    # Conversion to a 2-D numpy array of size (number of sentences, no: of features) 
    return np.array(vector_lists)
        
def vectorize_words(sentence, max_length=30):
    tokens = sentence.split()
    vector_size = None
    # vectorized_text consists of the feature vector representation of each word, 
    # after truncation and padding the sentence to max_length
    vectorized_text = []
    
    # Process each word in the sentence
    for word in tokens:
        try:
            vector = vocab.get_vector(word)
            if vector_size is None:
                vector_size = len(vector)
            vectorized_text.append(vector)
        except KeyError:
            continue
        
        # If we've reached the desired length, stop processing
        if len(vectorized_text) == max_length:
            break
 
    # Pad with zero vectors if needed
    while len(vectorized_text) < max_length:
        vectorized_text.append(np.zeros(vector_size))
    
    # Flatten the list of vectors into a single vector
    return np.array(vectorized_text).flatten()

def get_test_dataset(dataset):
    X = dataset['normalized_sentence'].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(dataset['relation_type'].values)
    
    return X, y
    
def collate_fn(batch):
    texts, labels = zip(*batch)
    
    max_len = max(text.size(0) for text in texts)
    
    padded_texts = torch.zeros(len(texts), max_len, texts[0].shape[1])
    for i, text in enumerate(texts):
        padded_texts[i, max_len - text.size(0):] = text
    
    labels = torch.stack(labels)
    return padded_texts, labels

def generate_loader(dataset, texts, labels, vocab, batch_size=32, shuffle=True):
    dataset = RelationDataset(texts, labels, vocab)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader

def create_dataset(dataset):
    # Tokenization and Encoding
    texts = dataset['normalized_sentence'].values
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(dataset['relation_type'].values)

    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val, label_encoder

def load_test_data(test_dataset, vocab, label_encoder):
    texts = test_dataset['sentence_text'].values
    labels = label_encoder.transform(test_dataset['relation_type'].values)
    
    test_data = RelationDataset(texts, labels, vocab)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
    return test_loader

# Vocab with word2vec
vocab = Vocabulary(word2vec)



