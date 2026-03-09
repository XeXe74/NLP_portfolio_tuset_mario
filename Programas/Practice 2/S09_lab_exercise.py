from typing import Any
import pandas as pd
import torch
import torch.nn as nn
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader

MAX_LEN = 64 # Maximum sequence length
BATCH_SIZE = 64 # Batch size for training
EMBED_DIM  = 128   # Size of each word embedding vector
HIDDEN_DIM = 256   # Number of LSTM hidden units per direction
NUM_LAYERS = 2     # Number of stacked LSTM layers
DROPOUT    = 0.4   # Dropout probability to reduce overfitting
NUM_CLASSES = 3    # Bearish, Bullish, Neutral

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
    
    
class TweetDataset(Dataset):
    """
    Function for creating a custom dataset for the tweets, which will be used for training and validation.
    """
    def __init__(self, df):
        self.samples = []
        
        # Iterate through each row in the DataFrame
        for _, row in df.iterrows():
            tokens = row['clean'].split()[:MAX_LEN] # Tokenize the cleaned text and limit to MAX_LEN
            ids    = [word2idx.get(t, UNK_IDX) for t in tokens] # Convert tokens to indices, using UNK_IDX for unknown tokens
            ids   += [PAD_IDX] * (MAX_LEN - len(ids)) # Pad the sequence with PAD_IDX to ensure it has a length of MAX_LEN
            self.samples.append((ids, int(row['label']))) # Store the token indices and label as a tuple

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns the input and label for a given index in the dataset.
        """
        ids, label = self.samples[idx]
        return (torch.tensor(ids,   dtype=torch.long),
                torch.tensor(label, dtype=torch.long))
    
    
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

# print(f"Vocabulary: {VOCAB_SIZE} tokens")

train_loader = DataLoader(TweetDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(TweetDataset(valid_df), batch_size=BATCH_SIZE, shuffle=False)

# print(f"Train batches: {len(train_loader)}")
# print(f"Valid batches: {len(valid_loader)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LSTM(
    vocab_size  = VOCAB_SIZE,
    embed_dim   = EMBED_DIM,
    hidden_dim  = HIDDEN_DIM,
    num_layers  = NUM_LAYERS,
    num_classes = NUM_CLASSES,
    dropout     = DROPOUT,
    pad_idx     = PAD_IDX
).to(device)

print(f"Device: {device}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")