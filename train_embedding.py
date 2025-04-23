from datasets import load_dataset
from transformers import BertTokenizer
import torch
from config import CHECKPOINT_DIR, SAVED_MODEL_DIR, CACHE_DIR
import os

dataset = load_dataset("imdb")

train_texts = dataset["train"]["text"][:10000]  
train_labels = dataset["train"]["label"][:10000]

test_texts = dataset["test"]["text"][:1000]  
test_labels = dataset["test"]["label"][:1000]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR)

def preprocess(texts, labels):
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return tokens["input_ids"], tokens["attention_mask"], torch.tensor(labels)

train_input_ids, train_attention_mask, train_labels = preprocess(train_texts, train_labels)
test_input_ids, test_attention_mask, test_labels = preprocess(test_texts, test_labels)

print("Data processing finished!")
print("train dataset embedding shape:", train_input_ids.shape)

from transformers import BertModel

bert_model = BertModel.from_pretrained("bert-base-uncased", cache_dir=CACHE_DIR).cuda()

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
            all_embeddings.append(outputs.last_hidden_state.cpu())
    
    return torch.cat(all_embeddings, dim=0)  # concat batch result

# train_embeddings = get_bert_embedding(train_input_ids, train_attention_mask)
# test_embeddings = get_bert_embedding(test_input_ids, test_attention_mask)
train_embeddings = get_bert_embedding(train_input_ids, train_attention_mask, batch_size=16)
test_embeddings = get_bert_embedding(test_input_ids, test_attention_mask, batch_size=16)

print("BERT embedding computed!")
print("train dataset embedding shape:", train_embeddings.shape)  # (batch_size, seq_len, hidden_dim)
torch.save(train_embeddings, "train_embeddings.pt")
torch.save(test_embeddings, "test_embeddings.pt")
torch.save(train_labels, "train_labels.pt")
torch.save(test_labels, "test_labels.pt")
print("Embedding file saved!")