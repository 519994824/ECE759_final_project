import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from config import CHECKPOINT_DIR, SAVED_MODEL_DIR, CACHE_DIR
import os
import torch

train_embeddings = torch.load("train_embeddings.pt")
test_embeddings = torch.load("test_embeddings.pt")
train_labels = torch.load("train_labels.pt")
test_labels = torch.load("test_labels.pt")
print("Embedding 加载完成!")

class SimpleTransformer(nn.Module):
    def __init__(self, d_model, heads):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        # QKV 计算
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 多头注意力后的输出层
        self.fc_out = nn.Linear(d_model, d_model)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # 计算 Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 分割多头
        Q = Q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数 Q * K^T / sqrt(d)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 计算加权 V
        attn_output = torch.matmul(attn_weights, V)

        # 还原维度
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # 通过输出层
        attn_output = self.fc_out(attn_output)

        # 残差连接 & 归一化
        x = self.norm1(attn_output + x)

        # 前馈网络 & 残差连接
        x = self.norm2(self.ffn(x) + x)

        return x

class TransformerClassifier(nn.Module):
    def __init__(self, d_model=768, heads=8, num_classes=2):
        super(TransformerClassifier, self).__init__()
        self.transformer = SimpleTransformer(d_model=d_model, heads=heads)
        self.fc = nn.Linear(d_model, num_classes)  # 分类层
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 池化

    def forward(self, x):
        x = self.transformer(x)
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)  # 池化到 (batch, d_model)
        return self.fc(x)  # 输出 logits

# 创建模型
model = TransformerClassifier().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 转换数据到 GPU
train_embeddings = train_embeddings.cuda()
train_labels = train_labels.cuda()
test_embeddings = test_embeddings.cuda()
test_labels = test_labels.cuda()

# 训练循环
epochs = 10  # 例如训练 100 轮
batch_size = 32
num_batches = len(train_embeddings) // batch_size

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
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # 每 20 轮保存一次 checkpoint
    if (epoch + 1) % 20 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

# 训练结束后保存最终模型
final_model_path = os.path.join(SAVED_MODEL_DIR, "final_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved at {final_model_path}")
    
model.eval()
with torch.no_grad():
    predictions = model(test_embeddings)
    predicted_labels = torch.argmax(predictions, dim=1)

accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)
print(f"测试集准确率: {accuracy:.4f}")