import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import pickle
from model import LSTMRelationClassifier
from data_vec import RelationDataset, collate_fn
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def load_test_data(test_data_path, vocab, label_encoder):
    test_dataset = pd.read_csv(test_data_path)
    texts = test_dataset['sentence_text'].values
    labels = label_encoder.transform(test_dataset['relation_type'].values)
    
    test_data = RelationDataset(texts, labels, vocab)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collate_fn)
    return test_loader

def main_test(test_data_path='test_dataset_dataframe.csv'):
    # Load vocab and label_encoder
    vocab_path = 'vocab.pkl'
    label_encoder_path = 'label_encoder.pkl'
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path = 'lstm_model/checkpoints/best_model.pth'
    model = LSTMRelationClassifier(embedding_dim=300,
                                   hidden_dim=128, 
                                   output_dim=len(label_encoder.classes_)).to(device)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    model.eval()

    test_loader = load_test_data(test_data_path, vocab, label_encoder)

    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions) * 100
    f1_scores = f1_score(all_labels, all_predictions, average=None)

    print(f'Accuracy on test data: {accuracy}%')
    for i, score in enumerate(f1_scores):
        class_name = label_encoder.inverse_transform([i])[0]
        print(f'F1 score for class {class_name}: {score}')

    return accuracy, f1_scores

if __name__ == "__main__":
    main_test()
