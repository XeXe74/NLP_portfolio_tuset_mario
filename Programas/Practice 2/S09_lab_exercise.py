from typing import Any
import pandas as pd
import torch
import torch.nn as nn
import re
from collections import Counter

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
    
    
def preprocess(text):
    """
    Function to preprocess the input text.
    """
    text = str(text).lower() # Convert to lowercase
    text = re.sub(r'http\S+|www\S+', '', text) # Remove URLs
    text = re.sub(r'@\w+', '', text) # Remove mentions
    text = re.sub(r'\$[A-Za-z]+', '', text) # Remove stock symbols
    text = re.sub(r'#', '', text) # Remove hashtags
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    
    return text


# Load the training and validation datasets
train_df = pd.read_csv('sent_train.csv')
valid_df = pd.read_csv('sent_valid.csv')

# print(f"Train: {len(train_df)} muestras")
# print(f"Valid: {len(valid_df)} muestras")
# print(train_df['label'].value_counts().sort_index())

# Preprocess the text data
train_df['clean'] = train_df['text'].apply(preprocess)
valid_df['clean']  = valid_df['text'].apply(preprocess)

# print(train_df[['text', 'clean']].head(3))

# Keep only tokens that appear at least 2 times
MIN_FREQ = 2

# Tokenize and build vocabulary from the training data
all_tokens = [tok for text in train_df['clean'] for tok in text.split()]

# Counting the frequency of each token in the training data
counter = Counter(all_tokens)

# <PAD> and <UNK> to handle padding and unknown tokens, respectively
vocab    = ['<PAD>', '<UNK>'] + [w for w, c in counter.items() if c >= MIN_FREQ]
word2idx = {w: i for i, w in enumerate(vocab)} # Mapping from word to index needed because LSTMs work with indices, not raw text

PAD_IDX   = word2idx['<PAD>']   # 0
UNK_IDX   = word2idx['<UNK>']   # 1
VOCAB_SIZE = len(vocab)

print(f"Vocabulary: {VOCAB_SIZE} tokens")