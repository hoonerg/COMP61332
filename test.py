from sklearn.metrics import classification_report
from dataset.dataset_svm import vectorize_sentence, get_test_dataset
import torch
from dataset.dataset_lstm import load_test_data, UserInputDataset
                            
from torch.utils.data import DataLoader
from train import trained_model_pickle_file
import pandas as pd
import os
import pickle
from config.model import LSTMRelationClassifier
from sklearn.metrics import f1_score, accuracy_score

def predict(model_type=None, user_input=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    new_path = os.path.join(os.getcwd(), "dataset/DDICorpus/Test/test_for_ddi_extraction_task/DrugBank/")
    # df = get_dataset_dataframe(directory=os.path.expanduser('~/ddi/dataset/DDICorpus/Test/test_for_ddi_extraction_task/DrugBank/'))
    dataset_path = 'dataset/test_dataset_dataframe.csv'
    df = pd.read_csv(dataset_path)
    
    if model_type is None:
        model_type = "LSTM"

    if model_type == "SVM":
        print("Infering SVM model...")
    
        X_test,Y = get_test_dataset(df)
        X = vectorize_sentence(X_test, 40)
        
        model = pd.read_pickle(trained_model_pickle_file)
        
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

            accuracy = accuracy_score(all_labels, all_predictions) * 100
            f1_scores = f1_score(all_labels, all_predictions, average=None)

            print(f'Accuracy on test data: {accuracy}%')
            for i, score in enumerate(f1_scores):
                class_name = label_encoder.inverse_transform([i])[0]
                print(f'F1 score for class {class_name}: {score}')

            return accuracy, f1_scores

    else:
        print(f"{model_type} is not supported.")
