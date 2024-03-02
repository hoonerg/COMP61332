import torch.nn as nn

# Model Definition
class LSTMRelationClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(LSTMRelationClassifier, self).__init__()
        # Removed the embedding layer as we will use pre-trained embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # Since 'text' is already embedded, we feed it directly to the LSTM
        _, (hidden, _) = self.lstm(text)
        # Squeeze the hidden state to remove the batch dimension and pass it to the fully connected layer
        return self.fc(hidden.squeeze(0))
