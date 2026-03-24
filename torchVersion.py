import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from collections import Counter

class SkipGramPyTorch(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # W_in and W_out become Embedding layers
        self.W_in = nn.Embedding(vocab_size, embed_dim)
        self.W_out = nn.Embedding(vocab_size, embed_dim)

        # Initialize embeddings with small random numbers
        nn.init.normal_(self.W_in.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.W_out.weight, mean=0.0, std=0.01)

    def forward(self, target, context, negatives):
        v_t = self.W_in(target)      # [batch_size, embed_dim]
        v_c = self.W_out(context)    # [batch_size, embed_dim]
        v_n = self.W_out(negatives)  # [batch_size, k, embed_dim]

        # --- POSITIVE PAIR ---
        score_pos = torch.sum(v_t * v_c, dim=1)
        pos_loss = -F.logsigmoid(score_pos)

        # --- NEGATIVE SAMPLES ---
        score_neg = torch.bmm(v_n, v_t.unsqueeze(2)).squeeze(2) # [batch_size, k]
        neg_loss = -F.logsigmoid(-score_neg).sum(dim=1)

        return (pos_loss + neg_loss).sum()

def train_pytorch_model(positive_pairs, unigram_tensor, vocab_size, embed_dim=100, epochs=20, neg_samples=5, lr=0.025, batch_size=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PyTorch model on {device}...")

    model = SkipGramPyTorch(vocab_size, embed_dim).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Prepare DataLoaders
    targets = torch.tensor([p[0] for p in positive_pairs], dtype=torch.long)
    contexts = torch.tensor([p[1] for p in positive_pairs], dtype=torch.long)
    dataset = TensorDataset(targets, contexts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 

    unigram_tensor = unigram_tensor.to(device)
    losses = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch_targets, batch_contexts in dataloader:
            batch_targets = batch_targets.to(device)
            batch_contexts = batch_contexts.to(device)

            # Sample negatives
            negatives = torch.multinomial(
                unigram_tensor, 
                len(batch_targets) * neg_samples, 
                replacement=True
            ).view(len(batch_targets), neg_samples)

            optimizer.zero_grad()
            
            # Forward & Loss
            loss = model(batch_targets, batch_contexts, negatives)
            
            # Backward & Update
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"PyTorch Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
        losses.append(total_loss)

    return losses


def load_text(dpath):
    text = open(dpath).read().lower()
    table = str.maketrans('', '', ",:;!?")
    text = text.translate(table)
    tokens = text.split()
    return tokens


def prepare_data(tokens, window_size=2):
    print("Preparing vocabulary and unigram distribution...")
    
    # 1. Build Vocabulary
    word_counts = Counter(tokens)
    unique_words = list(word_counts.keys())
    word2Ind = {word: i for i, word in enumerate(unique_words)}
    vocab_size = len(unique_words)
    
    # 2. Build Unigram Distribution (using pure PyTorch, no NumPy)
    counts = torch.tensor([word_counts[w] for w in unique_words], dtype=torch.float)
    unigram_tensor = counts.pow(0.75)
    unigram_tensor = unigram_tensor / unigram_tensor.sum()
    
    # 3. Compute Positive Pairs
    print("Extracting positive pairs...")
    positive_pairs = []
    token_indices = [word2Ind[w] for w in tokens]
    
    for i, target_idx in enumerate(token_indices):
        start = max(0, i - window_size)
        end = min(len(token_indices), i + window_size + 1)
        
        for j in range(start, end):
            if i != j: # Don't pair the word with itself
                context_idx = token_indices[j]
                positive_pairs.append((target_idx, context_idx))
                
    return positive_pairs, unigram_tensor, vocab_size


def main():
    dpath = "text8/text8"
    
    try:
        tokens = load_text(dpath)
    except FileNotFoundError:
        print(f"Error: Could not find '{dpath}'. Please check your file path.")
        return
        
    # Use 10,000 tokens as in your original code
    tokens = tokens[:10000]

    # Hyperparameters
    embed_dim = 100
    epochs = 20
    neg_samples = 5
    lr = 0.025
    batch_size = 1024 

    # 1. Prepare Data natively
    positive_pairs, unigram_tensor, vocab_size = prepare_data(tokens, window_size=2)

    # 2. Train PyTorch Model
    print("\n=== Starting PyTorch Training ===")
    pt_losses = train_pytorch_model(
        positive_pairs=positive_pairs,
        unigram_tensor=unigram_tensor,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        epochs=epochs,
        neg_samples=neg_samples,
        lr=lr,
        batch_size=batch_size 
    )

    # 3. Plot PyTorch Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), pt_losses, label='PyTorch Loss', marker='x', linewidth=2, color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Total Loss')
    plt.title('PyTorch SkipGram Training Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()