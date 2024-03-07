import pandas as pd
from nltk.corpus import stopwords
from test import predict

pd.set_option('display.width', 1000)

def normalize_sentence_by_char_offset(sentence_text, entity_e1_text, entity_e2_text, stop_words):
    """
    Normalize the sentence by replacing entity occurrences with placeholders and removing stopwords.
    """
    entities_to_replace = [
        {"text": entity_e1_text, "placeholder": " DRUG "},
        {"text": entity_e2_text, "placeholder": " OTHER_DRUG "}
    ]

    for entity in entities_to_replace:
        sentence_text = sentence_text.replace(entity["text"], entity["placeholder"])

    # Remove stopwords
    words = sentence_text.split()
    filtered_sentence = ' '.join(word for word in words if word.lower() not in stop_words).strip()
    
    return filtered_sentence

def get_user_input():
    """
    Get user input for sentence and entities, and normalize the input.
    """
    sentence = input("Enter the sentence: ")
    model_type = input("Enter Model Type (SVM/LSTM):")
    first_entity = input("Enter the first entity: ")
    second_entity = input("Enter the second entity: ")

    stop_words = set(stopwords.words('english')) | set(['the', ',', '-'])

    normalized_sentence = normalize_sentence_by_char_offset(sentence, first_entity, second_entity, stop_words)

    formatted_input = f'"{sentence}", "{first_entity}", "{second_entity}", "{normalized_sentence}"'
    return formatted_input, normalized_sentence, model_type

if __name__ == "__main__":
    user_input, normalized_sentence, model_type = get_user_input()
    
    print("\nPredicting with ",model_type," model...")
    if model_type == "SVM":
        predict("SVM", user_input, normalized_sentence)
    elif model_type == "LSTM":
        predict("LSTM", user_input)
        
