import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import pickle
from model import LSTMRelationClassifier
from data import RelationDataset, collate_fn  # Adjust according to your project structure

# Load vocab and label_encoder
vocab_path = '/Users/hoonchoi/main/Uni/Text_Mining/new_version/vocab.pkl'
label_encoder_path = '/Users/hoonchoi/main/Uni/Text_Mining/new_version/label_encoder.pkl'
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Assuming you have a function to load and preprocess your test data similar to your training data
# For example:
def load_test_data(test_data_path, vocab, label_encoder):
    test_dataset = pd.read_csv(test_data_path)
    texts = test_dataset['sentence_text'].values
    labels = label_encoder.transform(test_dataset['relation_type'].values)
    
    # Convert texts and labels into a dataset
    test_data = RelationDataset(texts, labels, vocab)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
    return test_loader

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_save_path = '/Users/hoonchoi/main/Uni/Text_Mining/new_version/best_model.pth'
model = LSTMRelationClassifier(len(vocab), embedding_dim=100, hidden_dim=128, output_dim=len(label_encoder.classes_))
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval()

# Load test data
test_data_path = '/Users/hoonchoi/main/Uni/Text_Mining/new_version/test_dataset_dataframe.csv'
test_loader = load_test_data(test_data_path, vocab, label_encoder)  # Ensure vocab and label_encoder are properly initialized

# Evaluate the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for texts, labels in test_loader:
        texts, labels = texts.to(device), labels.to(device)
        outputs = model(texts)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test data: {100 * correct / total}%')
