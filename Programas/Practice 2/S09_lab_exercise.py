from typing import Any
import pandas as pd
import torch
import torch.nn as nn
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MIN_FREQ = 2 # Minimum frequency for a token to be included in the vocabulary
MAX_LEN = 64 # Maximum sequence length
BATCH_SIZE = 64 # Batch size for training
EMBED_DIM = 128 # Size of each word embedding vector
HIDDEN_DIM = 256 # Number of LSTM hidden units per direction
NUM_LAYERS = 2 # Number of stacked LSTM layers
DROPOUT = 0.4 # Dropout probability to reduce overfitting
NUM_CLASSES = 3 # Bearish, Bullish, Neutral
EPOCHS = 7 # Number of training epochs

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

def train_one_epoch(model, loader):
    """
    Trains the model for one epoch on the given data loader and returns the average loss and accuracy.
    """
    model.train()  # set model to training mode
    total_loss = correct = total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad() # Clear gradients from the previous step
        logits = model(X) # Forward pass to get predictions
        loss   = criterion(logits, y) # Compute loss
        loss.backward() # Backpropagate the loss to compute gradients

        # Gradient clipping to prevent exploding gradients, which can occur in LSTMs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step() # Update weights

        # Accumulate loss and compute accuracy
        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)

    return total_loss / total, correct / total

def evaluate(model, loader):
    """
    Evaluates the model on the given data loader and returns the average loss, accuracy, and predictions.
    """
    model.eval() # Set model to evaluation mode
    total_loss = correct = total = 0
    all_preds, all_true = [], []

    with torch.no_grad():  # Disable gradient computation to save memory
        for X, y in loader:
            X, y = X.to(device), y.to(device) # Move data to the same device as the model

            logits = model(X) # Forward pass to get predictions
            loss = criterion(logits, y) # Compute loss

            # Accumulate loss and compute accuracy
            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_true


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

# print(f"Device: {device}")
# print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Calculate class weights based on the frequency of each class in the training data to handle class imbalance
counts  = train_df['label'].value_counts().sort_index().values
weights = torch.tensor(1.0 / counts, dtype=torch.float)
weights = (weights / weights.sum()).to(device)

# CrossEntropyLoss for multi-class classification, with class weights to handle class imbalance
criterion = nn.CrossEntropyLoss(weight=weights)

# Adam optimizer with weight decay for regularization to prevent overfitting
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Scheduler to reduce the learning rate when the validation loss plateaus, which can help improve convergence
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2
)

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_loss = float('inf')
best_state = None

print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
print("=" * 55)

# Training loop over a specified number of epochs
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader)
    vl_loss, vl_acc, val_preds, val_true = evaluate(model, valid_loader)

    # Adjust learning rate if val_loss stagnates
    scheduler.step(vl_loss)

    # Save metrics for plotting later
    history['train_loss'].append(tr_loss)
    history['train_acc'].append(tr_acc)
    history['val_loss'].append(vl_loss)
    history['val_acc'].append(vl_acc)

    # Save best model checkpoint based on validation loss
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"{epoch:>6} {tr_loss:>11.4f} {tr_acc:>10.4f} {vl_loss:>10.4f} {vl_acc:>9.4f}")

# Load best model before final evaluation
model.load_state_dict(best_state)

# Final evaluation using the best model checkpoint
_, _, final_preds, final_true = evaluate(model, valid_loader)

label_names = ['Bearish', 'Bullish', 'Neutral']

# Classification Report
print("\nClassification Report:")
print(classification_report(final_true, final_preds, target_names=label_names))

# Confusion matrix 
cm = confusion_matrix(final_true, final_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names, yticklabels=label_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()