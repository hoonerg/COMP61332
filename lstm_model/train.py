import torch
import torch.nn as nn
import torch.optim as optim
from data_vec import train_loader, val_loader, vocab, label_encoder
from model import LSTMRelationClassifier
from utils import EarlyStopping
import os
import pickle

# Define the train function
def train(model, train_loader, optimizer, criterion, device):
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
    return total_loss / len(train_loader)

# Define the evaluate function
def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for texts, labels in val_loader:
            texts, labels = texts.to(device), labels.to(device)
            output = model(texts)
            loss = criterion(output, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Function to save model checkpoint
def save_checkpoint(state, filename="model_checkpoint.pth.tar", checkpoint_dir="lstm_model/checkpoints/"):
    torch.save(state, os.path.join(checkpoint_dir, filename))

# Function to load model checkpoint
def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Main function to run the training process
def run_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, optimizer, and criterion
    model = LSTMRelationClassifier(embedding_dim=300, hidden_dim=128, output_dim=len(label_encoder.classes_)).to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    model_save_path = 'lstm_model/checkpoints/best_model.pth'
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, verbose=True, path=model_save_path)

    num_epochs = 300

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
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
