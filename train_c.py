import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.cpp_extension import load
from config import CHECKPOINT_DIR, SAVED_MODEL_DIR, CACHE_DIR, EPOCH, BATCH_SIZE
import math

train_embeddings = torch.load("train_embeddings.pt").cuda()
test_embeddings = torch.load("test_embeddings.pt").cuda()
train_labels = torch.load("train_labels.pt").cuda()
test_labels = torch.load("test_labels.pt").cuda()
print("Embedding load!")

my_cuda = load(
    name="my_cuda",
    sources=["my_cuda_kernel.cu"],
    extra_cuda_cflags=["-O3"],
)

# my_cuda = load(name="my_cuda", sources=["my_cuda_kernel_simple.cu"], verbose=True)

class MatMulBatchedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        # A, B: [B', M, K], [B', K, N] flattened already if you like
        ctx.save_for_backward(A, B)
        return my_cuda.matmul_batched(A, B)

    @staticmethod
    def backward(ctx, grad_C):
        A, B = ctx.saved_tensors
        if hasattr(my_cuda, "matmul_batched_backward"):
            grad_A, grad_B = my_cuda.matmul_batched_backward(grad_C.contiguous(), A, B)
        else:
            grad_A = torch.bmm(grad_C, B.transpose(-2, -1))
            grad_B = torch.bmm(A.transpose(-2, -1), grad_C)
        return grad_A, grad_B

class SoftmaxBatchedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        out = my_cuda.softmax_batched(X)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: [B*H, S, S]
        (softmax_out,) = ctx.saved_tensors
        sum_dim = torch.sum(grad_out * softmax_out, dim=-1, keepdim=True)
        grad_input = softmax_out * (grad_out - sum_dim)
        return grad_input


class SimpleTransformerCUDA(nn.Module):
    def __init__(self, d_model, heads):
        super(SimpleTransformerCUDA, self).__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        self.W_q    = nn.Parameter(torch.randn(d_model, d_model).cuda())
        self.W_k    = nn.Parameter(torch.randn(d_model, d_model).cuda())
        self.W_v    = nn.Parameter(torch.randn(d_model, d_model).cuda())
        self.fc_out = nn.Parameter(torch.randn(d_model, d_model).cuda())

        # GEMM
        # self.cuda_bmm = my_cuda.matmul_batched
        
        self.matmul = MatMulBatchedFunction.apply
        self.softmax = SoftmaxBatchedFunction.apply
        
    
    def softmax_batched(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert scores.dim() == 4 and dim in (-1, 3), \
            "wrong parameter"
        
        B, H, S, _ = scores.shape
        scores_flat = scores.reshape(B*H, S, S)
        # out_flat = my_cuda.softmax_batched(scores_flat)
        out_flat = self.softmax(scores_flat)
        return out_flat.view(B, H, S, S)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape

        Q = torch.matmul(x, self.W_q)  # [B, S, D]
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)

        # reshape [B*H, S, head_dim] and  [B*H, head_dim, S]
        Qh = (
            Q
            .view(batch_size, seq_len, self.heads, self.head_dim)
            .permute(0,2,1,3)
            .reshape(batch_size * self.heads, seq_len, self.head_dim)
        )
        Kh = (
            K
            .view(batch_size, seq_len, self.heads, self.head_dim)
            .permute(0,2,3,1)
            .reshape(batch_size * self.heads, self.head_dim, seq_len)
        )
        # batched matmul + zoom in
        # scores = self.cuda_bmm(Qh, Kh) / math.sqrt(self.head_dim)  
        
        scores = self.matmul(Qh, Kh) / math.sqrt(self.head_dim)  
        
        
        # recover [B, H, S, S]
        scores = scores.view(batch_size, self.heads, seq_len, seq_len)

        # attn = torch.softmax(scores, dim=-1)  # [B, H, S, S]
        attn = self.softmax_batched(scores, dim=-1)

        # flatten to [B*H, S, S]
        attn_flat = attn.reshape(batch_size * self.heads, seq_len, seq_len)
        Vh = (
            V
            .view(batch_size, seq_len, self.heads, self.head_dim)
            .permute(0,2,1,3)
            .reshape(batch_size * self.heads, seq_len, self.head_dim)
        )
        # head_out = self.cuda_bmm(attn_flat, Vh)  # [B*H, S, head_dim]
        
        head_out = self.matmul(attn_flat, Vh)

        # reshape [B, S, D]
        head_out = (
            head_out
            .view(batch_size, self.heads, seq_len, self.head_dim)
            .permute(0,2,1,3)
            .reshape(batch_size, seq_len, d_model)
        )

        out = torch.matmul(head_out, self.fc_out)  # [B, S, D]
        return out

class TransformerClassifierCUDA(nn.Module):
    def __init__(self, d_model=768, heads=8, num_classes=2):
        super(TransformerClassifierCUDA, self).__init__()
        self.transformer = SimpleTransformerCUDA(d_model=d_model, heads=heads)
        self.fc = nn.Linear(d_model, num_classes).cuda()
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = self.transformer(x)
        # [batch, seq_len, d_model] -> [batch, d_model]
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)
        return self.fc(x)

model = TransformerClassifierCUDA().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = EPOCH
batch_size = BATCH_SIZE
num_batches = len(train_embeddings) // batch_size

start_time = time.time()
print("Start training...")
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for i in tqdm(range(num_batches)):
        batch_x = train_embeddings[i * batch_size:(i + 1) * batch_size]
        batch_y = train_labels[i * batch_size:(i + 1) * batch_size]
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
overall_time = time.time() - start_time
print(f"Training finished, time cost: {overall_time:.2f} s")

model.eval()
with torch.no_grad():
    predictions = model(test_embeddings)
    predicted_labels = torch.argmax(predictions, dim=1)
accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)
print(f"Test dataset accuracy: {accuracy:.4f}")