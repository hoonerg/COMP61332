# -*- coding: utf-8 -*-
import glob
import os
from pyexpat import ExpatError
from xml.dom import minidom

import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm

STOP_WORDS = set(stopwords.words('english')) | set('the')

# pd.set_option('display.width', 1000)
dataset_csv_file = 'dataset_dataframe.csv'
types = set()

training_dataset_dataframe = None

def get_entity_dict(sentence_dom):
    entities = sentence_dom.getElementsByTagName('entity')
    entity_dict = {}
    for entity in entities:
        id = entity.getAttribute('id')
        text = entity.getAttribute('text')
        type = entity.getAttribute('type')
        charOffset = entity.getAttribute('charOffset')
        entity_dict[id] = {'text': text, 'type': type, 'charOffset': charOffset}
    return entity_dict

def replace_entities(sentence, entities):
    sorted_entities = sorted(entities.values(), key=lambda x: int(x['charOffset'].split('-')[0]))
    new_sentence = sentence
    for i, entity in enumerate(sorted_entities):
        try:
            start, end = map(int, entity['charOffset'].split('-'))
        except:
            start, end = map(int, entity['charOffset'].split(';'))
        replacement = entity['type'] if i % 2 == 0 else "Other_" + entity['type']
        new_sentence = new_sentence[:start] + replacement + new_sentence[end+1:]
    return new_sentence

def get_dataset_dataframe(directory=None):
    global training_dataset_dataframe, dataset_csv_file, types

    if directory is None:
        directory = os.path.expanduser('dataset/DDICorpus/Train/DrugBank/')

    dataset_csv_file_prefix = str(directory.split('/')[-3]).lower() + '_'
    dataset_csv_file = dataset_csv_file_prefix + dataset_csv_file

    if os.path.isfile(dataset_csv_file):
        return pd.read_csv(dataset_csv_file)

    lol = []
    total_files_to_read = glob.glob(directory + '*.xml')
    for file in tqdm(total_files_to_read):
        try:
            DOMTree = minidom.parse(file)
            sentences = DOMTree.getElementsByTagName('sentence')

            for sentence_dom in sentences:
                entity_dict = get_entity_dict(sentence_dom)
                sentence_text = sentence_dom.getAttribute('text')
                normalized_sentence = replace_entities(sentence_text, entity_dict)

                pairs = sentence_dom.getElementsByTagName('pair')
                for pair in pairs:
                    ddi_flag = pair.getAttribute('ddi')
                    if ddi_flag == 'true':
                        e1 = entity_dict[pair.getAttribute('e1')]['text']
                        e2 = entity_dict[pair.getAttribute('e2')]['text']
                        relation_type = pair.getAttribute('type')
                        lol.append([normalized_sentence, e1, e2, relation_type])
        except ExpatError:
            continue

    df = pd.DataFrame(lol, columns='normalized_sentence,e1,e2,relation_type'.split(','))
    df.to_csv(dataset_csv_file)
    return df

def get_training_label(row):
    global types

    types = pd.read_pickle('types')
    types = [t for t in types if t]
    type_list = list(types)
    relation_type = row.relation_type
    X = [i for i, t in enumerate(type_list) if relation_type == t]
    # s = np.sum(X)
    if X:
        return X[0]
    else:
        return 1

if __name__ == "__main__":
    directory = "dataset/DDICorpus/Train/DrugBank/"
    df = get_dataset_dataframe(directory)
    print(df.head())