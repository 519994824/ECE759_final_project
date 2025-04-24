import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from config import CHECKPOINT_DIR, SAVED_MODEL_DIR, CACHE_DIR, EPOCH, BATCH_SIZE
import os
import torch
import time
import math

class SimpleTransformer(nn.Module):
    def __init__(self, d_model, heads):
        super(SimpleTransformer, self).__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        self.W_q = nn.Parameter(torch.randn(d_model, d_model))
        self.W_k = nn.Parameter(torch.randn(d_model, d_model))
        self.W_v = nn.Parameter(torch.randn(d_model, d_model))
        self.fc_out = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape

        Q = torch.matmul(x, self.W_q)  # [B, S, D]
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)

        # reshape [B*H, S, head_dim] and [B*H, head_dim, S]
        Qh = (
            Q
            .view(batch_size, seq_len, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.heads, seq_len, self.head_dim)
        )
        Kh = (
            K
            .view(batch_size, seq_len, self.heads, self.head_dim)
            .permute(0, 2, 3, 1)
            .reshape(batch_size * self.heads, self.head_dim, seq_len)
        )
        
        scores = torch.bmm(Qh, Kh) / math.sqrt(self.head_dim)  # [B*H, S, S]
        
        # reshape [B, H, S, S]
        scores = scores.view(batch_size, self.heads, seq_len, seq_len)

        attn = F.softmax(scores, dim=-1)  # [B, H, S, S]

        # flatten to [B*H, S, S]
        attn_flat = attn.reshape(batch_size * self.heads, seq_len, seq_len)
        Vh = (
            V
            .view(batch_size, seq_len, self.heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.heads, seq_len, self.head_dim)
        )
        
        head_out = torch.bmm(attn_flat, Vh)  # [B*H, S, head_dim]

        # reshape [B, S, D]
        head_out = (
            head_out
            .view(batch_size, self.heads, seq_len, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, d_model)
        )

        out = torch.matmul(head_out, self.fc_out)  # [B, S, D]
        return out

class TransformerClassifier(nn.Module):
    def __init__(self, d_model=768, heads=8, num_classes=2):
        super(TransformerClassifier, self).__init__()
        self.transformer = SimpleTransformer(d_model=d_model, heads=heads)
        self.fc = nn.Linear(d_model, num_classes)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.transformer(x)
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)
        return self.fc(x)

train_embeddings = torch.load("train_embeddings.pt").contiguous().cuda()
test_embeddings = torch.load("test_embeddings.pt").contiguous().cuda()
train_labels = torch.load("train_labels.pt").cuda()
test_labels = torch.load("test_labels.pt").cuda()

model = TransformerClassifier().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = EPOCH
batch_size = BATCH_SIZE
num_batches = len(train_embeddings) // batch_size

print("Start training...")
start_time = time.time()
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for i in tqdm(range(num_batches)):
        batch_x = train_embeddings[i * batch_size : (i + 1) * batch_size]
        batch_y = train_labels  [i * batch_size : (i + 1) * batch_size]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

overall_time = time.time() - start_time
print(f"Training finished, time cost: {overall_time:.2f} s")

model.eval()
with torch.no_grad():
    predictions = model(test_embeddings)
    predicted_labels = torch.argmax(predictions, dim=1)

accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)
print(f"Test accuracy: {accuracy:.4f}")