import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.cpp_extension import load
from config import CHECKPOINT_DIR, SAVED_MODEL_DIR, CACHE_DIR, EPOCH
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

# def cuda_matmul(A, B):
#     return my_cuda.matmul_cuda(A, B)

# class MatMulCUDAFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, A, B):
#         C = my_cuda.matmul_forward(A, B)
#         ctx.save_for_backward(A, B)
#         return C

#     @staticmethod
#     def backward(ctx, grad_out):
#         A, B = ctx.saved_tensors
#         grad_A, grad_B = my_cuda.matmul_backward(
#             grad_out.contiguous(), A, B
#         )
#         return grad_A, grad_B

# def cuda_matmul(A, B):
#     return MatMulCUDAFunction.apply(A, B)

class SoftmaxCUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = torch.empty_like(x)
        my_cuda.softmax_forward(x, out)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (softmax_out,) = ctx.saved_tensors
        grad_input = my_cuda.softmax_backward(grad_out.contiguous(), softmax_out)
        return grad_input

def cuda_softmax(x):
    return SoftmaxCUDAFunction.apply(x)

# class SimpleTransformerCUDA(nn.Module):
#     def __init__(self, d_model, heads):
#         super(SimpleTransformerCUDA, self).__init__()
#         self.d_model = d_model
#         self.heads = heads
#         self.head_dim = d_model // heads

#         self.W_q = nn.Parameter(torch.randn(d_model, d_model).cuda())
#         self.W_k = nn.Parameter(torch.randn(d_model, d_model).cuda())
#         self.W_v = nn.Parameter(torch.randn(d_model, d_model).cuda())
#         self.fc_out = nn.Parameter(torch.randn(d_model, d_model).cuda())

#     def forward(self, x):
#         # x: [batch_size, seq_len, d_model]
#         batch_size, seq_len, d_model = x.shape

#         Q = torch.matmul(x, self.W_q)  # [batch, seq_len, d_model]
#         K = torch.matmul(x, self.W_k)
#         V = torch.matmul(x, self.W_v)

#         outputs = []
#         for b in range(batch_size):
#             # multi head，reshape [heads, seq_len, head_dim]
#             Q_b = Q[b].view(seq_len, self.heads, self.head_dim).permute(1, 0, 2)
#             K_b = K[b].view(seq_len, self.heads, self.head_dim).permute(1, 0, 2)
#             V_b = V[b].view(seq_len, self.heads, self.head_dim).permute(1, 0, 2)
            
#             head_outputs = []
#             for h in range(self.heads):
#                 q = Q_b[h]  # [seq_len, head_dim]
#                 k = K_b[h]  # [seq_len, head_dim]
#                 scores = cuda_matmul(q, k.t()) / (self.head_dim ** 0.5)
#                 attn_weights = torch.softmax(scores, dim=-1)
#                 head_out = cuda_matmul(attn_weights, V_b[h])
#                 head_outputs.append(head_out)
#             concat = torch.cat(head_outputs, dim=-1)
#             out = cuda_matmul(concat, self.fc_out)
#             outputs.append(out)
#         return torch.stack(outputs)  # [batch_size, seq_len, d_model]

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
        self.cuda_bmm = my_cuda.matmul_batched
    
    def softmax_batched(self, scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert scores.dim() == 4 and dim in (-1, 3), \
            "wrong parameter"
        
        B, H, S, _ = scores.shape
        scores_flat = scores.reshape(B*H, S, S)
        out_flat = my_cuda.softmax_batched(scores_flat)
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
        scores = self.cuda_bmm(Qh, Kh) / math.sqrt(self.head_dim)  
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
        head_out = self.cuda_bmm(attn_flat, Vh)  # [B*H, S, head_dim]

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
batch_size = 32
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