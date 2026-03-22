import numpy as np
import matplotlib.pyplot as plt


def load_text(dpath):
    """
    Load, clean and tokenize the dataset.

    Input
    -----
    dpath : path to text file

    Output
    ------
    tokens : list of words
    """
    text = open(dpath).read().lower()
    table = str.maketrans('', '', ",:;!?")
    text = text.translate(table)
    tokens = text.split()
    #optional if want to work on a sample
    #tokens = tokens[ :10000]

    return tokens


class SkipGram:

    def __init__(self, tokens, embedDim):
        """
        Initialize SkipGram model

        Parameters
        ----------
        tokens : list of words in corpus
        embedDim : embedding dimension
        """

        self.tokens = tokens
        self.embedDim = embedDim

        self.word2Ind = None #mapping from word to index
        self.unigram = None #prob of each word

        self.indices = None # contains indexes of the words contained in tokens
        self.positive_pairs = []

        self.W_in = None # encoding matrix
        self.W_out = None # decoding matrix


    def createIndexUnigram(self, unigram_parameter=0.75):
        """
        Build vocabulary and unigram distribution.

        Steps
        -----
        1. count frequency of each word
        2. map each word to an index
        3. build unigram distribution for negative sampling
        """

        frequency = {}

        for token in self.tokens:
            currOcc = frequency.get(token, 0)
            frequency[token] = currOcc + 1

        self.word2Ind = {w: idx for idx, w in enumerate(frequency.keys())}

        vocab_size = len(self.word2Ind)

        self.unigram = np.zeros(vocab_size)

        for word, idx in self.word2Ind.items():
            self.unigram[idx] = frequency[word] ** unigram_parameter

        self.unigram = self.unigram / np.sum(self.unigram)

        # convert tokens → indices once
        self.indices = [self.word2Ind[w] for w in self.tokens]


    def compute_positive(self, window_size=2):
        """
        Compute all positive (target, context) pairs using a sliding window.

        Example
        -------
        sentence:  the cat eats fish
        window=1
        pairs:
        (cat,the)
        (cat,eats)
        """

        N = len(self.indices) # == len(self.tokens)

        for i in range(N):

            target = self.indices[i]

            left = max(0, i - window_size)
            right = min(N, i + window_size + 1)

            for j in range(left, right):
                if j == i:
                    continue
                context = self.indices[j]
                self.positive_pairs.append((target, context))


    def init_embeddings(self):
        """
        Initialize input and output embedding matrices.
        """

        vocab_size = len(self.word2Ind)

        self.W_in = np.random.randn(vocab_size, self.embedDim) * 0.01
        self.W_out = np.random.randn(vocab_size, self.embedDim) * 0.01


    def sample_negative(self, k):
        """
        Sample k negative words from unigram distribution.
        """

        vocab_size = len(self.word2Ind)

        negatives = np.random.choice(
            vocab_size,
            size=k,
            p=self.unigram
        )
        return negatives




    def sigmoid(self, x):
        # Ensure x is a numpy array for element-wise operations
        x = np.asanyarray(x)
        
        # Initialize output array
        res = np.empty_like(x, dtype=float)
        
        # Mask for positive and negative values
        pos_mask = (x >= 0)
        neg_mask = ~pos_mask
        
        # For x >= 0: 1 / (1 + exp(-x))
        # -x is negative, so exp(-x) is <= 1. Stable.
        res[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        
        # For x < 0: exp(x) / (1 + exp(x))
        # x is negative, so exp(x) is <= 1. Stable.
        exp_x = np.exp(x[neg_mask])
        res[neg_mask] = exp_x / (1 + exp_x)
        
        return res


    def log_sigmoid(self, x):
        """
        Numerically stable log(sigmoid(x))
        """
        # Uses the identity: log(1/(1+exp(-x))) = -log(1+exp(-x))
        # To stabilize: -max(0, -x) - log(1 + exp(-abs(x)))
        return np.where(x > 0, -np.log(1 + np.exp(-x)), x - np.log(1 + np.exp(x)))


    def train(self, epochs=1, neg_samples=5, lr=0.025):
        losses = []
        for epoch in range(epochs):
            total_loss = 0.0

            for target, context in self.positive_pairs:
                v_t = self.W_in[target]
                v_c = self.W_out[context]
                
                # --- POSITIVE PAIR ---
                score_pos = np.dot(v_t, v_c)
                
                # Calcolo Loss (Stabile)
                total_loss -= self.log_sigmoid(score_pos)
                
                # Calcolo Gradiente
                grad_score_pos = self.sigmoid(score_pos) - 1
                
                # Accumulatore per l'aggiornamento di v_t
                grad_v_t = grad_score_pos * v_c
                # Aggiornamento immediato del contesto
                self.W_out[context] -= lr * grad_score_pos * v_t
            
                # --- NEGATIVE SAMPLES ---
                negatives = self.sample_negative(neg_samples)
                for neg in negatives:
                    v_n = self.W_out[neg]
                    score_neg = np.dot(v_t, v_n)
                    
                    # Calcolo Loss (Stabile): -log(sigmoid(-score))
                    total_loss -= self.log_sigmoid(-score_neg)
                    
                    # Calcolo Gradiente per campione negativo
                    grad_score_neg = self.sigmoid(score_neg)
                    
                    grad_v_t += grad_score_neg * v_n
                    self.W_out[neg] -= lr * grad_score_neg * v_t

                # Aggiorna il target una sola volta per finestra
                self.W_in[target] -= lr * grad_v_t

            # Decadimento del learning rate
            lr *= 0.95 
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
            losses.append(total_loss)

        # Plot dei risultati
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), losses, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.show()


def main():

    dpath = "text8/text8"

    tokens = load_text(dpath)

    model = SkipGram(tokens, 100)

    model.createIndexUnigram()

    model.compute_positive(window_size=2)

    model.init_embeddings()

    model.train(
        epochs=20,
        neg_samples=5,
        lr=0.025
    )
    



if __name__ == "__main__":
    main()