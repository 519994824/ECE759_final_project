from datasets import load_dataset
from transformers import BertTokenizer
import torch
from config import CHECKPOINT_DIR, SAVED_MODEL_DIR, CACHE_DIR
import os

# 加载 IMDb 电影评论数据集
dataset = load_dataset("imdb")

# 只取一小部分训练数据（减少训练时间）
train_texts = dataset["train"]["text"][:10000]  
train_labels = dataset["train"]["label"][:10000]

test_texts = dataset["test"]["text"][:1000]  
test_labels = dataset["test"]["label"][:1000]

# 加载 BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR)

# 进行分词处理
def preprocess(texts, labels):
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return tokens["input_ids"], tokens["attention_mask"], torch.tensor(labels)

train_input_ids, train_attention_mask, train_labels = preprocess(train_texts, train_labels)
test_input_ids, test_attention_mask, test_labels = preprocess(test_texts, test_labels)

print("数据预处理完成!")
print("训练数据形状:", train_input_ids.shape)

from transformers import BertModel

# 加载 BERT 作为 embedding 计算器
bert_model = BertModel.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR).cuda()

# 计算 embedding
# def get_bert_embedding(input_ids, attention_mask):
#     input_ids = input_ids.cuda()
#     attention_mask = attention_mask.cuda()
#     with torch.no_grad():
#         outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
#         return outputs.last_hidden_state  # 取最后一层 hidden state

from torch.utils.data import DataLoader, TensorDataset

def get_bert_embedding(input_ids, attention_mask, batch_size=16):
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    all_embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            batch_input_ids, batch_attention_mask = batch
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()

            outputs = bert_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            all_embeddings.append(outputs.last_hidden_state.cpu())  # 先转回 CPU，减少 GPU 压力
    
    return torch.cat(all_embeddings, dim=0)  # 拼接所有 batch 的结果

# 获取训练集 & 测试集 embedding
# train_embeddings = get_bert_embedding(train_input_ids, train_attention_mask)
# test_embeddings = get_bert_embedding(test_input_ids, test_attention_mask)
train_embeddings = get_bert_embedding(train_input_ids, train_attention_mask, batch_size=16)
test_embeddings = get_bert_embedding(test_input_ids, test_attention_mask, batch_size=16)

print("BERT embedding 计算完成!")
print("训练集 embedding 形状:", train_embeddings.shape)  # (batch_size, seq_len, hidden_dim)
# 保存 embedding 到文件
torch.save(train_embeddings, "train_embeddings.pt")
torch.save(test_embeddings, "test_embeddings.pt")
torch.save(train_labels, "train_labels.pt")
torch.save(test_labels, "test_labels.pt")
print("Embedding 保存完成!")