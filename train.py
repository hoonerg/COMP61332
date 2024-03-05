# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations

from nltk import ngrams
# from sklearn.cross_validation import train_test_split
# In the new version, the train test split function 
# is moved into sklearn.model_selection
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from dataset.dataset_svm import extract_training_data_from_dataframe
from dataset.dataset_lstm import RelationDataset, create_dataset, \
                    find_max_word_length, generate_loader, vectorize_sentence, vocab
from config.model import LSTMRelationClassifier
from config.utils import EarlyStopping, evaluate
import os
import pickle
import numpy as np


trained_model_pickle_file = 'trained_model.pkl'

def main(model_type=None):
    import pandas as pd
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = 'dataset/train_dataset_dataframe.csv'
    df = pd.read_csv(dataset_path)

    if model_type is None:
        model_type = "LSTM"

    if model_type == "SVM":
        print("Training SVM model...")

        # Both X_train and y_train are numpy arrays   
        X_train_sentences, X_test_sentences, y_train, y_test, label_encoder = create_dataset(df)
        max_word_length = find_max_word_length(X_train_sentences)
        # Embedding the sentences       
        X_train = vectorize_sentence(X_train_sentences, max_word_length)
        X_test = vectorize_sentence(X_test_sentences ,max_word_length)

        model = SVC(kernel='linear')
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        pd.to_pickle(model, trained_model_pickle_file)
        print('Score : ', score)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

    elif model_type == "LSTM":
        print("Training LSTM model...")

        # Generate data loaders
        
        train_loader = generate_loader(df, X_train, y_train, vocab, batch_size=32, shuffle=True)
        val_loader = generate_loader(df, X_val, y_val, vocab, batch_size=32, shuffle=False)

        # Initialize model, optimizer, and criterion
        model = LSTMRelationClassifier(embedding_dim=300, hidden_dim=128, output_dim=len(label_encoder.classes_)).to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        model_save_path = 'results/checkpoints/lstm_best_model.pth'
        
        # Early stopping
        early_stopping = EarlyStopping(patience=15, verbose=True, path=model_save_path)

        num_epochs = 300

        for epoch in range(num_epochs):
            model.train()

            total_loss = 0
            for texts, labels in train_loader:
                texts, labels = texts.to(device), labels.to(device)
                
                optimizer.zero_grad()
                output = model(texts)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            val_loss = evaluate(model, val_loader, criterion, device)

            # Log the training and validation loss
            print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Save vocab and label encoder
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)

        with open('label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)        

    else:
        print(f"{model_type} is not supported.")