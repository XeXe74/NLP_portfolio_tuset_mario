from typing import Any
import pandas as pd
import torch
import torch.nn as nn

# Load the dataset
train = pd.read_csv('sent_train.csv')
test = pd.read_csv('sent_valid.csv')

class LSTM(nn.Module):
    """
    LSTM-based model for sentiment analysis.
    """
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout, pad_idx):
        super().__init__()

        # Embedding layer with padding index
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        Forward pass through the model.
        """
        embedded = self.dropout(self.embedding(x)) # Apply dropout to the embedded input

        _, (hidden, _) = self.lstm(embedded) # Get the hidden state from the LSTM

        final = torch.cat([hidden[-2], hidden[-1]], dim=1) # Concatenate the final forward and backward hidden states

        return self.fc(self.dropout(final)) 