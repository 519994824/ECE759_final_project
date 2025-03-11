import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.cpp_extension import load
from config import CHECKPOINT_DIR, SAVED_MODEL_DIR, CACHE_DIR

# 加载预先保存的 embedding 与 label，并转移到 GPU
train_embeddings = torch.load("train_embeddings.pt").cuda()
test_embeddings = torch.load("test_embeddings.pt").cuda()
train_labels = torch.load("train_labels.pt").cuda()
test_labels = torch.load("test_labels.pt").cuda()
print("Embedding 加载完成!")

# 加载自定义 CUDA 内核（编译并加载 my_cuda_kernel.cu 文件）
my_cuda = load(name="my_cuda", sources=["my_cuda_kernel.cu"], verbose=True)

def cuda_matmul(A, B):
    """
    使用自定义 CUDA 算子进行矩阵乘法。
    要求 A 和 B 均为 GPU 上的 float32 张量。
    """
    return my_cuda.matmul_cuda(A, B)

# 可直接使用 torch.softmax 进行 softmax 计算（GPU 版已足够高效）
def cuda_softmax(x):
    return torch.softmax(x, dim=-1)

###############################################
# 定义使用 CUDA 加速矩阵运算的 Transformer 模块
###############################################

class SimpleTransformerCUDA(nn.Module):
    def __init__(self, d_model, heads):
        super(SimpleTransformerCUDA, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        # 定义线性变换权重，注意将权重初始化在 GPU 上
        self.W_q = nn.Parameter(torch.randn(d_model, d_model).cuda())
        self.W_k = nn.Parameter(torch.randn(d_model, d_model).cuda())
        self.W_v = nn.Parameter(torch.randn(d_model, d_model).cuda())
        self.fc_out = nn.Parameter(torch.randn(d_model, d_model).cuda())

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]，已在 GPU 上
        batch_size, seq_len, d_model = x.shape

        # 使用内置矩阵乘法计算 Q, K, V（权重变换）
        Q = torch.matmul(x, self.W_q)  # [batch, seq_len, d_model]
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)

        outputs = []
        for b in range(batch_size):
            # 将单个样本拆分为多个 head，调整形状为 [heads, seq_len, head_dim]
            Q_b = Q[b].view(seq_len, self.heads, self.head_dim).permute(1, 0, 2)
            K_b = K[b].view(seq_len, self.heads, self.head_dim).permute(1, 0, 2)
            V_b = V[b].view(seq_len, self.heads, self.head_dim).permute(1, 0, 2)
            
            head_outputs = []
            for h in range(self.heads):
                q = Q_b[h]  # [seq_len, head_dim]
                k = K_b[h]  # [seq_len, head_dim]
                # 使用自定义 CUDA 算子计算注意力分数：q 与 k 的转置相乘，再除以 sqrt(head_dim)
                scores = cuda_matmul(q, k.t()) / (self.head_dim ** 0.5)
                # 使用 torch.softmax 计算注意力权重（GPU 上高效）
                attn_weights = torch.softmax(scores, dim=-1)
                # 使用自定义 CUDA 算子计算加权求和
                head_out = cuda_matmul(attn_weights, V_b[h])
                head_outputs.append(head_out)
            # 拼接各 head 输出，得到 [seq_len, d_model]
            concat = torch.cat(head_outputs, dim=-1)
            # 使用自定义 CUDA 算子经过输出层
            out = cuda_matmul(concat, self.fc_out)
            outputs.append(out)
        return torch.stack(outputs)  # [batch_size, seq_len, d_model]

class TransformerClassifierCUDA(nn.Module):
    def __init__(self, d_model=768, heads=8, num_classes=2):
        super(TransformerClassifierCUDA, self).__init__()
        self.transformer = SimpleTransformerCUDA(d_model=d_model, heads=heads)
        self.fc = nn.Linear(d_model, num_classes).cuda()  # 分类层
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 池化层

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]，已在 GPU 上
        x = self.transformer(x)
        # 池化：将 [batch, seq_len, d_model] 转换为 [batch, d_model]
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)
        return self.fc(x)

#####################################
# 使用 CUDA 模型训练与测试（基于自定义 CUDA 算子）
#####################################

model = TransformerClassifierCUDA().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 10
batch_size = 32
num_batches = len(train_embeddings) // batch_size

start_time = time.time()
print("开始训练 CUDA 模型（使用自定义 CUDA 算子）...")
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
print(f"训练完成，总用时: {overall_time:.2f} s")

# 测试模型
model.eval()
with torch.no_grad():
    predictions = model(test_embeddings)
    predicted_labels = torch.argmax(predictions, dim=1)
accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)
print(f"测试集准确率: {accuracy:.4f}")