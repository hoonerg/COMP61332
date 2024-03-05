# -*- coding: utf-8 -*-
from nltk.util import ngrams

from dataset.read_dataset import get_dataset_dataframe
from grammar.chunker import Chunker
from grammar.syntactic_grammar import PatternGrammar

frequent_word_pairs = None
K = 200
import pandas as pd

from spacy.lang.en import English

parser = English()
import os

from itertools import combinations
from collections import Counter


def get_dataset_dictionary():
    # Returns the most number of frequently occuring words in the CSV
    top_post_fixed_word_file = 'top_post_fixed_word.pkl'
    print("Top post fixed word: ", top_post_fixed_word_file)
    if os.path.isfile(top_post_fixed_word_file):
        return pd.read_pickle(top_post_fixed_word_file)
    
    df = get_dataset_dataframe()
    word_counter = Counter()
    for _, row in df.iterrows():
        # exclude duplicates in same line and sort to ensure one word is always before other
        unique_tokens = sorted(set(word for word in row.normalized_sentence.split()))
        # Creates bi-grams of all the tokens from the normalized sentence
        bi_grams = ngrams(row.normalized_sentence.split(), 2)
        # Example of bigrams: [("Concurrent_bf", "therapy_bf"), ("therapy_bf", "DRUG"), ("DRUG", "TNF_be"), 
        # ("TNF_be", "antagonists_be"), ("antagonists_be", "recommended_be"), ("recommended_be", "._be")]
        word_counter += Counter([' '.join(bi_gram).strip() for bi_gram in bi_grams])
            # Creates a list by splitting each bigram tuple and combines it with a space
            # Word counter will have: {
            #     'Concurrent_bf therapy_bf': 1,
            #     'therapy_bf DRUG': 1,
            #     'DRUG TNF_be': 1,
            #     'TNF_be antagonists_be': 1,
            #     'antagonists_be recommended_be': 1,
            #     'recommended_be ._be': 1
            # }
        word_counter += Counter(unique_tokens)
        # Adding count of all the unique tokens as well
    frequent_words = sorted(list(dict(word_counter.most_common(100000)).keys()))  
    # return the actual Counter object
    # which consists of the most frequently occuring tokens and bigrams
    pd.to_pickle(frequent_words, top_post_fixed_word_file)
    return frequent_words


def extract_top_word_pair_features():
    frequent_phrase_pickle_path = 'frequent_phrase.pkl'
    if not os.path.isfile(frequent_phrase_pickle_path):
        df = get_dataset_dataframe()
        pair_counter = Counter()
        

        for _, row in df.iterrows():
            unique_tokens = sorted(set(word for word in row.normalized_sentence.split()))
            # exclude duplicates in same line and sort to ensure one word is always before other
            combos = combinations(unique_tokens, 2)
            # Creates all possible combinations of unique tokens with each other
            # takign two at a time
            pair_counter += Counter(combos)

        print("PAIR COUNTER:- ", pair_counter)
        frequent_phrase = sorted(list(dict(pair_counter.most_common(K)).keys()))  
        # return the actual Counter object
        pd.to_pickle(frequent_phrase, frequent_phrase_pickle_path)
    else:
        frequent_phrase = pd.read_pickle(frequent_phrase_pickle_path)
    print('frequent_phrase: ' , frequent_phrase[:5])
    return frequent_phrase


def extract_top_syntactic_grammar_trio():
    top_syntactic_grammar_trio_file = 'top_syntactic_grammar_trio_file.pkl'
    if os.path.isfile(top_syntactic_grammar_trio_file):
        return pd.read_pickle(top_syntactic_grammar_trio_file)

    df = get_dataset_dataframe()
    trio_counter = Counter()
    for _, row in df.iterrows():
        combos = extract_syntactic_grammar(row.sentence_text)
        trio_counter += Counter(combos)

    frequent_trio_counter = sorted(list(dict(trio_counter.most_common(K)).keys()))  # return the actual Counter object
    pd.to_pickle(frequent_trio_counter, top_syntactic_grammar_trio_file)
    return frequent_trio_counter


def extract_dependency_relations(sentence):
    # TODO : introduce dependency relation later
    parsedEx = parser(sentence)
    for token in parsedEx:
        print(token.orth_, token.dep_, token.head.orth_)


def extract_syntactic_grammar(sentence):
    # Only using first grammar, returns a parser object
    grammar = PatternGrammar().get_syntactic_grammar(0)
    chunk_dict = Chunker(grammar).chunk_sentence(sentence)
    trigrams_list = []
    for key, pos_tagged_sentences in chunk_dict.items():
        pos_tags = [token[1] for pos_tagged_sentence in pos_tagged_sentences for token in pos_tagged_sentence]
        if len(pos_tags) > 2:
            trigrams = ngrams(pos_tags, 3)
            trigrams_list = [' '.join(trigram) for trigram in trigrams]

    return trigrams_list


# if __name__ == '__main__':
#     df = get_dataset_dataframe()
#     print(get_dataset_dictionary())
