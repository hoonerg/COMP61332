from sklearn.metrics import classification_report

from dataset.dataset_lstm import load_test_data
from training.train import extract_training_data_from_dataframe, trained_model_pickle_file
import pandas as pd
import os

def predict(model_type=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    new_path = os.path.join(os.getcwd(), "dataset/DDICorpus/Test/test_for_ddi_extraction_task/DrugBank/")
    # df = get_dataset_dataframe(directory=os.path.expanduser('~/ddi/dataset/DDICorpus/Test/test_for_ddi_extraction_task/DrugBank/'))
    dataset_path = 'dataset/test_dataset_dataframe.csv'
    df = pd.read_csv(dataset_path)
    
    if model_type is None:
        model_type = "LSTM"

    if model_type == "SVM":
        print("Infering SVM model...")

        X, Y = extract_training_data_from_dataframe(df)
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
