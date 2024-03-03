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
        print("TEXT", self.texts[item])
        vectorized_text = [self.vocab.get_vector(word) for word in self.texts[item].split()]
        vectorized_text = np.stack(vectorized_text)
        # print("VECTORIZED TEXT AND SHAPE:", vectorized_text, vectorized_text.shape)
        # print("VECTORIZED flattened:", vectorized_text.flatten())
        return torch.tensor(vectorized_text, dtype=torch.float), torch.tensor(self.labels[item], dtype=torch.long)

# X_train - texts
# y_train - labels
# df - dataset
# def vectorize_words(sentence):
#     tokens = sentence.split()
#     print("TEXT 2", sentence)
#     vectorized_text = [vocab.get_vector(word) for word in tokens]
#     vectorized_text = np.stack(vectorized_text)
#     print("VECTORIZED TEXT 2:", vectorized_text)
#     return vectorized_text.flatten()

def vectorize_words(sentence, max_length=30):
    """
    Vectorize a sentence, padding or truncating to ensure a fixed-size output.
    
    :param sentence: The sentence to vectorize.
    :param vocab: A vocabulary with a get_vector(word) method to obtain vectors.
    :param max_length: The fixed number of words to encode per sentence.
    :return: A flattened array representing the vectorized, padded/truncated sentence.
    """
    # Split the sentence into tokens
    tokens = sentence.split()
    vector_size = None
    # Initialize a list to hold the vectorized words
    vectorized_text = []
    
    # Process each word in the sentence
    for word in tokens:
        try:
            # Attempt to get the vector for the word
            vector = vocab.get_vector(word)
            if vector_size is None:
                vector_size = len(vector)
            vectorized_text.append(vector)
        except KeyError:
            # If the word is not in the vocabulary, you can choose to skip it
            # or append a zero vector; for now, let's skip it
            continue
        
        # If we've reached the desired length, stop processing
        if len(vectorized_text) == max_length:
            break
    
    # print("TESTING:- ", vectorized_text)       
    
    # Pad with zero vectors if needed
    while len(vectorized_text) < max_length:
        vectorized_text.append(np.zeros(vector_size))
    
    # Flatten the list of vectors into a single vector
    return np.array(vectorized_text).flatten()

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



