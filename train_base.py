# !!! all deprecated


# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# import torch.nn.functional as F
# from config import CHECKPOINT_DIR, SAVED_MODEL_DIR, CACHE_DIR, EPOCH
# import os
# import torch
# import time

# train_embeddings = torch.load("train_embeddings.pt")
# test_embeddings = torch.load("test_embeddings.pt")
# train_labels = torch.load("train_labels.pt")
# test_labels = torch.load("test_labels.pt")
# print("Embedding load!")

# class SimpleTransformer(nn.Module):
#     def __init__(self, d_model, heads):
#         super(SimpleTransformer, self).__init__()
#         self.d_model = d_model
#         self.heads = heads
#         self.head_dim = d_model // heads

#         self.W_q = nn.Linear(d_model, d_model)
#         self.W_k = nn.Linear(d_model, d_model)
#         self.W_v = nn.Linear(d_model, d_model)
        
#         self.fc_out = nn.Linear(d_model, d_model)

#         self.ffn = nn.Sequential(
#             nn.Linear(d_model, d_model * 4),
#             nn.ReLU(),
#             nn.Linear(d_model * 4, d_model),
#         )

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)

#     def forward(self, x):
#         batch_size, seq_len, d_model = x.shape

#         Q = self.W_q(x)
#         K = self.W_k(x)
#         V = self.W_v(x)
        
#         Q = Q.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
#         K = K.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
#         V = V.view(batch_size, seq_len, self.heads, self.head_dim).transpose(1, 2)
        
#         scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
#         attn_weights = F.softmax(scores, dim=-1)
#         attn_output = torch.matmul(attn_weights, V)
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
#         attn_output = self.fc_out(attn_output)
#         x = self.norm1(attn_output + x)
#         x = self.norm2(self.ffn(x) + x)

#         return x

# class TransformerClassifier(nn.Module):
#     def __init__(self, d_model=768, heads=8, num_classes=2):
#         super(TransformerClassifier, self).__init__()
#         self.transformer = SimpleTransformer(d_model=d_model, heads=heads)
#         self.fc = nn.Linear(d_model, num_classes)
#         self.pooling = nn.AdaptiveAvgPool1d(1)

#     def forward(self, x):
#         x = self.transformer(x)
#         x = self.pooling(x.transpose(1, 2)).squeeze(-1)
#         return self.fc(x)

# model = TransformerClassifier().cuda()
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# train_embeddings = train_embeddings.cuda()
# train_labels = train_labels.cuda()
# test_embeddings = test_embeddings.cuda()
# test_labels = test_labels.cuda()

# epochs = EPOCH
# batch_size = 32
# num_batches = len(train_embeddings) // batch_size
# print("Start training...")
# start_time = time.time()
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0.0
#     for i in tqdm(range(num_batches)):
#         batch_x = train_embeddings[i * batch_size:(i + 1) * batch_size]
#         batch_y = train_labels[i * batch_size:(i + 1) * batch_size]

#         optimizer.zero_grad()
#         outputs = model(batch_x)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
    
#     avg_loss = total_loss / num_batches
#     print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

#     if (epoch + 1) % 20 == 0:
#         checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pt")
#         torch.save({
#             'epoch': epoch + 1,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': avg_loss,
#         }, checkpoint_path)
#         print(f"Checkpoint saved at {checkpoint_path}")
# overall_time = time.time() - start_time
# print(f"Training finished, time cost: {overall_time:.2f} s")
# final_model_path = os.path.join(SAVED_MODEL_DIR, "final_model.pth")
# torch.save(model.state_dict(), final_model_path)
# print(f"Final model saved at {final_model_path}")
    
# model.eval()
# with torch.no_grad():
#     predictions = model(test_embeddings)
#     predicted_labels = torch.argmax(predictions, dim=1)

# accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)
# print(f"Test dataset accuracy: {accuracy:.4f}")