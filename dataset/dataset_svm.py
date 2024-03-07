import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dataset.dataset_lstm import Vocabulary
from gensim.models import KeyedVectors
import numpy as np

word2vec_path = 'GoogleNews-vectors-negative300.bin.gz'
word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
vocab = Vocabulary(word2vec)

def find_max_word_length(sentences):
    max_lengths = []
    for sentence in sentences:
        # Splitting the sentence into words and finding the length of each word
        word_lengths = [len(word) for word in sentence.split()]
        # Finding the maximum length in the current sentence
        max_length = max(word_lengths)
        max_lengths.append(max_length)
    return max(max_lengths)

def vectorize_sentence(sentences, vocab, max_length= 30):
    vector_lists = []
    
    for sentence in sentences:
        vector = vectorize_words(sentence, vocab, max_length)
        vector_lists.append(vector)
    
    # Conversion to a 2-D numpy array of size (number of sentences, no: of features) 
    return np.array(vector_lists)
        
def vectorize_words(sentence, vocab, max_length=30):
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

def get_test_dataset(dataset, label_encoder, vocab):
    # Returns embedded form for both X and y
    X_test = dataset['normalized_sentence'].values

    X = vectorize_sentence(X_test, vocab, 40)
    y = label_encoder.fit_transform(dataset['relation_type'].values)
 
    return X, y

def get_X_for_inference(normalized_sentence, vocab):
    X = vectorize_sentence([normalized_sentence], vocab, 40)

    return X
  