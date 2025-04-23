import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from config import CHECKPOINT_DIR, SAVED_MODEL_DIR, CACHE_DIR, EPOCH

train_embeddings = torch.load("train_embeddings.pt")
test_embeddings = torch.load("test_embeddings.pt")
train_labels = torch.load("train_labels.pt")
test_labels = torch.load("test_labels.pt")
print("Embedding load!")

torch.set_num_threads(2)

def numpy_matmul(A, B):
    """
    A: torch.Tensor, shape [M, K]
    B: torch.Tensor, shape [K, N]
    Return: torch.Tensor, shape [M, N]
    """
    A_np = A.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()
    C_np = np.dot(A_np, B_np)
    return torch.tensor(C_np, dtype=A.dtype)

def numpy_softmax(x):
    x_np = x.detach().cpu().numpy()
    exps = np.exp(x_np)
    sum_exps = np.sum(exps)
    return torch.tensor(exps / sum_exps, dtype=x.dtype)

class SimpleTransformerCPU(nn.Module):
    def __init__(self, d_model, heads):
        super(SimpleTransformerCPU, self).__init__()
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

        Q = x @ self.W_q  # [batch_size, seq_len, d_model]
        K = x @ self.W_k
        V = x @ self.W_v

        outputs = []
        for b in range(batch_size):
            # multi head: [seq_len, d_model] -> [heads, seq_len, head_dim]
            Q_b = Q[b].view(seq_len, self.heads, self.head_dim).permute(1, 0, 2)
            K_b = K[b].view(seq_len, self.heads, self.head_dim).permute(1, 0, 2)
            V_b = V[b].view(seq_len, self.heads, self.head_dim).permute(1, 0, 2)
            
            head_outputs = []
            for h in range(self.heads):
                q = Q_b[h]  # [seq_len, head_dim]
                k = K_b[h]  # [seq_len, head_dim]
                scores = numpy_matmul(q, k.t()) / (self.head_dim ** 0.5)
                attn_weights_list = []
                for row in scores:
                    attn_weights_list.append(numpy_softmax(row))
                attn_weights = torch.stack(attn_weights_list)  # [seq_len, seq_len]
                head_out = numpy_matmul(attn_weights, V_b[h])
                head_outputs.append(head_out)
            # concat head，get [seq_len, d_model]
            concat = torch.cat(head_outputs, dim=-1)
            out = numpy_matmul(concat, self.fc_out)
            outputs.append(out)
        return torch.stack(outputs)  # [batch_size, seq_len, d_model]

class TransformerClassifierCPU(nn.Module):
    def __init__(self, d_model=768, heads=8, num_classes=2):
        super(TransformerClassifierCPU, self).__init__()
        self.transformer = SimpleTransformerCPU(d_model=d_model, heads=heads)
        self.fc = nn.Linear(d_model, num_classes)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = self.transformer(x)
        # [batch, seq_len, d_model] -> [batch, d_model]
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)
        return self.fc(x)

model_cpu = TransformerClassifierCPU(d_model=768, heads=8, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cpu.parameters(), lr=1e-4)

epochs = EPOCH
batch_size = 32
num_batches = len(train_embeddings) // batch_size

start_time = time.time()
print("Start training...")
for epoch in range(epochs):
    model_cpu.train()
    total_loss = 0.0
    for i in tqdm(range(num_batches)):
        batch_x = train_embeddings[i * batch_size:(i + 1) * batch_size]
        batch_y = train_labels[i * batch_size:(i + 1) * batch_size]
        optimizer.zero_grad()
        outputs = model_cpu(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
total_time = time.time() - start_time
print(f"Training finished, time cost: {total_time:.2f} s")

model_cpu.eval()
with torch.no_grad():
    predictions = model_cpu(test_embeddings)
    predicted_labels = torch.argmax(predictions, dim=1)
accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)
print(f"Test dataset accuracy: {accuracy:.4f}")