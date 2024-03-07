from sklearn.metrics import classification_report
from dataset.dataset_svm import get_X_for_inference, vectorize_sentence, get_test_dataset
import torch
from dataset.dataset_lstm import load_test_data, UserInputDataset
                            
from torch.utils.data import DataLoader
from train import trained_model_pickle_file
import pandas as pd
import os
import pickle
from config.model import LSTMRelationClassifier
from sklearn.metrics import f1_score, accuracy_score

def predict(model_type=None, user_input=None, normalized_sentence= None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = 'dataset/test_dataset_dataframe.csv'
    df = pd.read_csv(dataset_path)
    
    if model_type is None:
        model_type = "LSTM"

    if model_type == "SVM":
        print("Infering SVM model...")
        
        model = pd.read_pickle(trained_model_pickle_file)
        
        #Load both vocab and label_encoder
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        if user_input:
            print("User Input :", user_input)
            X = get_X_for_inference(normalized_sentence, vocab)
            y_pred  = model.predict(X)
            
            predicted_label = label_encoder.inverse_transform(y_pred)[0]
            
            print("Predicted Label is:- ", predicted_label)            
        else:
            X,Y = get_test_dataset(df, label_encoder, vocab)   
            y_pred  = model.predict(X)
            
            print(classification_report(Y, y_pred))

    elif model_type == "LSTM":
        print("Infering LSTM model...")

        # Load vocab and label_encoder
        vocab_path = 'vocab.pkl'
        label_encoder_path = 'label_encoder.pkl'
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        model_save_path = 'results/checkpoints/lstm_best_model.pth'
        model = LSTMRelationClassifier(embedding_dim=300,
                                    hidden_dim=128, 
                                    output_dim=len(label_encoder.classes_)).to(device)
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.to(device)
        model.eval()

        if user_input:
            print("User input is given: ", user_input)
            user_input_dataset = UserInputDataset(user_input, vocab)
            user_input_loader = DataLoader(user_input_dataset, batch_size=1, shuffle=False)

            with torch.no_grad():
                for vectorized_text in user_input_loader:
                    vectorized_text = vectorized_text.to(device)
                    output = model(vectorized_text)
                    _, predicted = torch.max(output, 1)
                    predicted_label = label_encoder.inverse_transform([predicted.cpu().numpy()[0]])[0]
                    print(f'Predicted label for the provided input: {predicted_label}')
                    
                    return predicted_label

        else:
            test_loader = load_test_data(df, vocab, label_encoder)

            all_predictions = []
            all_labels = []
            with torch.no_grad():
                for texts, labels in test_loader:
                    texts, labels = texts.to(device), labels.to(device)
                    outputs = model(texts)
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            overall_f1_score = f1_score(all_labels, all_predictions, average='micro') * 100

            # Calculate F1 scores for each class without averaging
            f1_scores_by_class = f1_score(all_labels, all_predictions, average=None)

            # Print overall F1 score
            print(f'Overall F1 Score on test data: {overall_f1_score:.2f}%')

            # Print F1 score for each class
            for i, score in enumerate(f1_scores_by_class):
                class_name = label_encoder.inverse_transform([i])[0]
                print(f'F1 score for class {class_name}: {score:.4f}')

            return overall_f1_score, f1_scores_by_class

    else:
        print(f"{model_type} is not supported.")
