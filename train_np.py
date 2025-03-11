import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from config import CHECKPOINT_DIR, SAVED_MODEL_DIR, CACHE_DIR

# 加载预先保存好的 embedding 和 label
train_embeddings = torch.load("train_embeddings.pt")
test_embeddings = torch.load("test_embeddings.pt")
train_labels = torch.load("train_labels.pt")
test_labels = torch.load("test_labels.pt")
print("Embedding 加载完成!")

# 为避免多线程干扰，设置线程数（仅针对 PyTorch 内部操作）
torch.set_num_threads(2)

###############################################
# 使用 numpy 进行矩阵运算的实现
###############################################

def numpy_matmul(A, B):
    """
    使用 numpy 计算矩阵乘法
    A: torch.Tensor, shape [M, K]
    B: torch.Tensor, shape [K, N]
    返回: torch.Tensor, shape [M, N]
    """
    # 注意：为了保证计算正确，这里调用 .detach().cpu().numpy()
    A_np = A.detach().cpu().numpy()
    B_np = B.detach().cpu().numpy()
    C_np = np.dot(A_np, B_np)
    # 保持数据类型一致
    return torch.tensor(C_np, dtype=A.dtype)

def numpy_softmax(x):
    """
    使用 numpy 实现 softmax（针对 1D tensor）
    """
    x_np = x.detach().cpu().numpy()
    exps = np.exp(x_np)
    sum_exps = np.sum(exps)
    return torch.tensor(exps / sum_exps, dtype=x.dtype)

###############################################
# 定义使用 numpy 加速计算的 Transformer 模块
###############################################

class SimpleTransformerCPU(nn.Module):
    def __init__(self, d_model, heads):
        super(SimpleTransformerCPU, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        # 定义线性变换权重（参数）
        self.W_q = nn.Parameter(torch.randn(d_model, d_model))
        self.W_k = nn.Parameter(torch.randn(d_model, d_model))
        self.W_v = nn.Parameter(torch.randn(d_model, d_model))
        self.fc_out = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape

        # 内置矩阵乘法计算 Q, K, V
        Q = x @ self.W_q  # [batch_size, seq_len, d_model]
        K = x @ self.W_k
        V = x @ self.W_v

        outputs = []
        for b in range(batch_size):
            # 将单个样本拆分为多个 head：从 [seq_len, d_model] -> [heads, seq_len, head_dim]
            Q_b = Q[b].view(seq_len, self.heads, self.head_dim).permute(1, 0, 2)
            K_b = K[b].view(seq_len, self.heads, self.head_dim).permute(1, 0, 2)
            V_b = V[b].view(seq_len, self.heads, self.head_dim).permute(1, 0, 2)
            
            head_outputs = []
            for h in range(self.heads):
                q = Q_b[h]  # [seq_len, head_dim]
                k = K_b[h]  # [seq_len, head_dim]
                # 使用 numpy 实现注意力分数计算：q 与 k 的转置相乘，并除以 sqrt(head_dim)
                scores = numpy_matmul(q, k.t()) / (self.head_dim ** 0.5)
                # 对 scores 的每一行使用 numpy_softmax
                attn_weights_list = []
                for row in scores:
                    attn_weights_list.append(numpy_softmax(row))
                attn_weights = torch.stack(attn_weights_list)  # [seq_len, seq_len]
                # 使用 numpy 计算加权求和
                head_out = numpy_matmul(attn_weights, V_b[h])
                head_outputs.append(head_out)
            # 拼接各 head 输出，得到 [seq_len, d_model]
            concat = torch.cat(head_outputs, dim=-1)
            # 输出层：使用 numpy 实现矩阵乘法
            out = numpy_matmul(concat, self.fc_out)
            outputs.append(out)
        return torch.stack(outputs)  # [batch_size, seq_len, d_model]

class TransformerClassifierCPU(nn.Module):
    def __init__(self, d_model=768, heads=8, num_classes=2):
        super(TransformerClassifierCPU, self).__init__()
        self.transformer = SimpleTransformerCPU(d_model=d_model, heads=heads)
        self.fc = nn.Linear(d_model, num_classes)  # 分类层
        self.pooling = nn.AdaptiveAvgPool1d(1)      # 池化层

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = self.transformer(x)
        # 池化：将 [batch, seq_len, d_model] 转换为 [batch, d_model]
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)
        return self.fc(x)

#####################################
# 使用 CPU 模型训练与测试（基于 numpy 加速矩阵运算）
#####################################

# BERT embedding 已经在 CPU 上保存
model_cpu = TransformerClassifierCPU(d_model=768, heads=8, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_cpu.parameters(), lr=1e-4)

epochs = 10
batch_size = 32
num_batches = len(train_embeddings) // batch_size

start_time = time.time()
print("开始训练 CPU 模型（numpy加速矩阵运算）...")
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
print(f"训练完成，总用时: {total_time:.2f} s")

# 测试模型
model_cpu.eval()
with torch.no_grad():
    predictions = model_cpu(test_embeddings)
    predicted_labels = torch.argmax(predictions, dim=1)
accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)
print(f"测试集准确率: {accuracy:.4f}")