import torch.nn as nn

class TransactionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)  
        self.fc1 = nn.Linear(embed_dim, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)  # Averaging token embeddings
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x