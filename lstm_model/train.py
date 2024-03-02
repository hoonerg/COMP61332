import torch
import torch.nn as nn
import torch.optim as optim
from data_vec import X_train, X_val, y_train, y_val, vocab, label_encoder, generate_loader  # Ensure this import works
from model import LSTMRelationClassifier  # Ensure this import works
from utils import EarlyStopping  # Importing EarlyStopping
import os

# Training
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize data loaders
train_loader = generate_loader(X_train, y_train, vocab, batch_size=32, shuffle=True)
val_loader = generate_loader(X_val, y_val, vocab, batch_size=32, shuffle=False)

# Initialize model, optimizer, and criterion
print(len(vocab))
model = LSTMRelationClassifier(embedding_dim=300,
                               hidden_dim=128, 
                               output_dim=len(label_encoder.classes_)).to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    torch.save(state, os.path.join(checkpoint_dir, filename))

def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Implement early stopping
model_save_path = 'lstm_model/checkpoints/best_model.pth'

# Early stopping
early_stopping = EarlyStopping(patience=10, verbose=True, path=model_save_path)

# Adjust the number of epochs as needed
num_epochs = 200

checkpoint_dir = 'lstm_model/checkpoints/'

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
    
    # Log the training and validation loss
    print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    # Check early stopping condition
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break


import pickle

# Assuming vocab and label_encoder are your objects
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)