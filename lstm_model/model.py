import torch.nn as nn

class LSTMRelationClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super(LSTMRelationClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        _, (hidden, _) = self.lstm(text)
        return self.fc(hidden.squeeze(0))
