import math
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Embedding(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.dim = dim

    def forward(self, x):
        return self.embedding(x) * np.sqrt(self.dim)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - x_mean) / (x_std + self.eps) + self.beta

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.Wo = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k ,v, mask=None):
        batch_size, _, dim = q.shape
        head_dim = dim / self.heads
        Q = self.Wq(q)
        K = self.Wk(k)
        V = self.Wv(v)

        # batch_size * headers * seq_len * header_dim
        Q = Q.view(batch_size, -1, self.headers, head_dim).contiguous().permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.headers, head_dim).contiguous().permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.headers, head_dim).contiguous().permute(0, 2, 1, 3)

        # batch_size * headers * seq_len * seq_len
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        attn = torch.matmul(attn, V)

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.Wo(attn)

class EncoderLayer(nn.Module):
    def __init__(self, dim, headers, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dim, headers, dropout)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, dim),
            self.dropout
        )
        self.ln1 = LayerNorm(dim)
        self.ln2 = LayerNorm(dim)

    def forward(self, x, src_mask=None):
        ln = self.ln1(x)
        x = x + self.dropout(self.attention(ln, ln, ln, mask=src_mask))
        x = x + self.ffn(self.ln2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, dim, layers, headers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(dim, headers, dropout) for _ in range(layers)])
        self.ln = LayerNorm(dim)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=None)
        return self.ln(x)

class DecoderLayer(nn.Module):
    def __init__(self, dim, headers, dropout=0.1):
        super().__init__()
        self.mask_attention = MultiHeadAttention(dim, headers, dropout)
        self.ln1 = LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, headers, dropout)
        self.ln2 = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ln3 = LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Linear(dim*4, dim),
            self.dropout
        )
    
    def forward(self, encode, decode, src_mask=None, tgt_mask=None):
        ln_de = self.ln1(decode)
        x = decode + self.dropout(self.mask_attention(ln_de, ln_de, ln_de, mask=tgt_mask))
        ln_de2 = self.ln2(x)
        x = x + self.dropout(self.attention(ln_de2, encode, encode, mask=src_mask))
        x = x + self.ffn(self.ln3(x))
        return x

class Decoder(nn.Module):
    def __init__(self, dim, layers, headers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(dim, headers, dropout) for _ in range(layers)])
        self.ln = LayerNorm(dim)

    def forward(self, encode, decode, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            decode = layer(encode, decode, src_mask, tgt_mask)
        return self.ln(decode)
        
class minTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, dim, layers, headers, dropout=0.1):
        super().__init__()
        self.embedding_src = Embedding(src_vocab, dim)
        self.embedding_tgt = Embedding(tgt_vocab, dim)
        self.encoder = Encoder(dim, layers, headers, dropout)
        self.decoder = Decoder(dim, layers, headers, dropout)
        self.linear = nn.Linear(dim, tgt_vocab)

    def forward(self, src, tgt):
        src_emb = self.embedding_src(src)
        tgt_emb = self.embedding_tgt(tgt)
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        encode = self.encoder(src_emb, src_mask)
        decode = self.decoder(encode, tgt_emb, src_mask, tgt_mask)
        return self.linear(decode)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones(1, seq_len, seq_len)).bool()
        tgt_mask = tgt_mask & nopeak_mask  # 结合padding和look-ahead mask
        return src_mask, tgt_mask

if __name__ == '__main__':

    dim = 512
    layer = 6
    headers = 8
    dropout = 0.1
    epochs = 10
    batch_size = 4

    model = minTransformer(100, 100, dim, layer, headers, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            src, tgt = batch
            src = src.to(device)
            tgt = tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")