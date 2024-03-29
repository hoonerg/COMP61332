import glob
from xml.dom import minidom
import pandas as pd
from nltk.corpus import stopwords
from tqdm import tqdm

def get_entity_by_id(entities, entity_id):
    """
    Retrieve a specific entity by its ID from the list of entities.
    """
    for entity in entities:
        if entity['id'] == entity_id:
            return entity
    return None

def normalize_sentence_by_char_offset(sentence_text, entity_e1, entity_e2, stop_words):
    # Remove commas and hyphens from the sentence text
    sentence_text = sentence_text.replace(',', ' ').replace('-', ' ')

    entities_to_replace = [entity_e1, entity_e2]
    # Sort entities by start position to handle them in correct order
    entities_to_replace.sort(key=lambda x: x['start'])

    offset_diff = 0
    for i, entity in enumerate(entities_to_replace):
        # Determine the suffix based on the entity's order
        suffix = '_first ' if i == 0 else '_second '
        # The replacement text now appends the suffix to the entity text
        replacement_text = entity['text'] + suffix
        
        # Calculate the start and end positions of the entity within the current version of the sentence
        start = entity['start'] + offset_diff
        # The end position should not be incremented by 1 here, since we're replacing the entire entity text
        end = entity['end'] + offset_diff + 1
        
        # Ensure spacing around replacements if not at the beginning or end of the sentence
        if start > 0 and sentence_text[start-1].isalnum():
            replacement_text = ' ' + replacement_text
        if end < len(sentence_text) and sentence_text[end].isalnum():
            replacement_text = replacement_text + ' '
        
        # Replace the entity text with the modified replacement_text in the sentence
        sentence_text = sentence_text[:start] + replacement_text + sentence_text[end:]
        
        # Adjust offset_diff based on the length difference between the original entity text and the replacement text
        offset_diff += len(replacement_text) - (end - start)


    # Remove stopwords after replacements
    words = sentence_text.split()
    filtered_sentence = ' '.join(word for word in words if word.lower() not in stop_words).strip()
    
    return filtered_sentence

def get_entities_with_char_offset(sentence_dom):
    # Identify entities based on 'charOffset' from xml dataset
    entities = sentence_dom.getElementsByTagName('entity')
    entity_list = []
    for entity in entities:
        id = entity.getAttribute('id')
        word = entity.getAttribute('text')
        char_offsets = entity.getAttribute('charOffset').split(';')
        for char_range in char_offsets:
            start, end = map(int, char_range.split('-'))
            entity_list.append({'id': id, 'text': word, 'start': start, 'end': end, 'charOffset': char_range})
    return entity_list

def get_dataset_dataframe(directory, dataset_csv_file, types):
    # Identify entities and sentence from original xml dataset
    stop_words = set(stopwords.words('english')) | set(['the'])

    data_records = []
    total_files_to_read = glob.glob(directory + '*.xml')
    print('total_files_to_read:', len(total_files_to_read), 'from dir:', directory)
    for file in tqdm(total_files_to_read):
        try:
            DOMTree = minidom.parse(file)
            sentences = DOMTree.getElementsByTagName('sentence')

            for sentence_dom in sentences:
                entities = get_entities_with_char_offset(sentence_dom)
                entity_dict = {e['id']: e for e in entities}
                pairs = sentence_dom.getElementsByTagName('pair')
                sentence_text = sentence_dom.getAttribute('text')
                for pair in pairs:
                    ddi_flag = pair.getAttribute('ddi')
                    if ddi_flag == 'true':
                        e1_id = pair.getAttribute('e1')
                        e2_id = pair.getAttribute('e2')
                        e1_entity = entity_dict.get(e1_id)
                        e2_entity = entity_dict.get(e2_id)
                        relation_type = pair.getAttribute('type')
                        types.add(relation_type)
                        normalized_sentence = normalize_sentence_by_char_offset(sentence_text, e1_entity, e2_entity, stop_words)
                        data_records.append([sentence_text, e1_entity['text'], e2_entity['text'], relation_type, normalized_sentence])
        except Exception as e:
            pass#print(f"Error processing file {file}: {e}")

    df = pd.DataFrame(data_records, columns='sentence_text,e1_text,e2_text,relation_type,normalized_sentence'.split(','))
    df.to_csv(dataset_csv_file)
    return df, types

if __name__ == "__main__":
    types = set()
    directory_1 = "dataset/DDICorpus/Train/merged_version/" # Path to original dataset
    dataset_csv_file = 'dataset/train_dataset_dataframe.csv'
    df1, types = get_dataset_dataframe(directory_1, dataset_csv_file, types)
    
    directory_2 = "dataset/DDICorpus/Test/test_for_ddi_extraction_task/merged_version/"
    dataset_csv_file = 'dataset/test_dataset_dataframe.csv'
    df2, types = get_dataset_dataframe(directory_2, dataset_csv_file, types)

    pd.to_pickle(types, 'types.pkl')
    print(df1.shape)
    print(df2.shape)