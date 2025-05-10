import pandas as pd
import torch
from torch_geometric.data import Data
from collections import defaultdict
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
import argparse
from pathlib import Path
import random
import numpy


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--saved_model_path")
args = parser.parse_args()

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, vector_emb_dim, graph_type):
        super().__init__()
        if graph_type == "GCN":
            print("Building GCNConv model")
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
        elif graph_type == "SAGE":
            print("Building SAGEConv model")
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        else:
            print("Building GATConv model")
            heads = args.heads
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
            self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)
       
        self.linear = nn.Linear(out_channels, vector_emb_dim)
        self.relu = nn.ReLU()
        self.to(device)

    def forward(self, graph):
        x, edge_index = graph.x.to(device), graph.edge_index.to(device)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.linear(x)
        x = self.relu(x)
        return x  # Node embeddings: [num_nodes, D]

class TripleEmbedder(nn.Module):
    def __init__(self, node_embed_dim, num_rels):
        super().__init__()
        self.rel_embed = nn.Embedding(num_rels, node_embed_dim)

    def forward(self, head_ids, rel_ids, tail_ids, node_embeddings):
        h = node_embeddings[head_ids]
        r = self.rel_embed(torch.tensor(rel_ids)).to(device)
        t = node_embeddings[tail_ids]
        triple_embed = h + r + t  # Simple additive scoring
        return triple_embed  # [batch, D]

class QueryEncoder(nn.Module):
    def __init__(self, model_name=args.query_embedding_model):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, query_texts):
        encoded = self.tokenizer(query_texts, return_tensors="pt", padding="longest")
        out = self.bert(**encoded.to(device))
        return out.last_hidden_state[:, 0, :]  # [CLS] token

class TestGraphDataset(Dataset):
    def __init__(self, df):
        self.questions = df.Question.tolist()
        self.gold_triples = df.Gold_Triples.tolist()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question_emb = query_encoder(self.questions[idx])
        positive_triples = self.gold_triples[idx].tolist()
        return question_emb, positive_triples

gcn = torch.load(f"{args.saved_model_path}/gcn.pt", weights_only=False)
query_encoder = torch.load(f"{args.saved_model_path}/query_encoder.pt", weights_only=False)
triple_encoder = torch.load(f"{args.saved_model_path}/triple_encoder.pt", weights_only=False)

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')

test_df = pd.read_parquet("../../data/mined_data/test_gold.parquet")

gcn.eval()
query_encoder.eval()
triple_encoder.eval()

full_graph = pd.read_csv("../../data/mined_data/full_graph.csv")
triples = 



hard_hits_5 = []
soft_hits_5 = []

hard_hits_10 = []
soft_hits_10 = []

hard_hits_15 = []
soft_hits_15 = []

recall_5 = []
mrr = []

for row in tqdm(test_df.itertuples()):
	with torch.no_grad():
		node_embeddings = gcn(graph)  # [num_nodes, D]
        query_emb = batch[0].squeeze(1)  # [B, D]

        all_triple_embeds = triple_encoder(
            all_head_ids, all_rel_ids, all_tail_ids, node_embeddings
        )  # [N_triples, D]

        query_emb = nn.functional.normalize(query_emb, dim=-1)
        all_triple_embeds = nn.functional.normalize(all_triple_embeds, dim=-1)
        
        sims = torch.matmul(query_emb, all_triple_embeds.T)
        
        top5 = torch.topk(sims, k=5)
        top5_indices = top5.indices.tolist()
        for idx, index_list in enumerate(top5_indices):
          predictions = [] 
          for index in index_list:
            predictions.append((H[index], R[index], T[index]))
          hard_hits_5.append(hard_hits(predictions, batch[1][idx]))
          soft_hits_5.append(soft_hits(predictions, batch[1][idx]))
          recall_5.append(recall(predictions, batch[1][idx]))
          mrr.append(mrr_calc(predictions, batch[1][idx]))

        top10 = torch.topk(sims, k=10)
        top10_indices = top10.indices.tolist()
        for idx, index_list in enumerate(top10_indices):
          predictions = []
          for index in index_list:
            predictions.append((H[index], R[index], T[index]))
          hard_hits_10.append(hard_hits(predictions, batch[1][idx]))
          soft_hits_10.append(soft_hits(predictions, batch[1][idx]))

        top15 = torch.topk(sims, k=15)
        top15_indices = top15.indices.tolist()
        for idx, index_list in enumerate(top15_indices):
          predictions = []
          for index in index_list:
            predictions.append((H[index], R[index], T[index]))
          hard_hits_15.append(hard_hits(predictions, batch[1][idx]))
          soft_hits_15.append(soft_hits(predictions, batch[1][idx]))

print(f"Hard hits@5: {sum(hard_hits_5)/len(hard_hits_5):.2f}")
print(f"Soft hits@5: {sum(soft_hits_5)/len(soft_hits_5):.2f}")

print(f"Hard hits@10: {sum(hard_hits_10)/len(hard_hits_10):.2f}")
print(f"Soft hits@10: {sum(soft_hits_10)/len(soft_hits_10):.2f}")

print(f"Hard hits@15: {sum(hard_hits_15)/len(hard_hits_15):.2f}")
print(f"Soft hits@15: {sum(soft_hits_15)/len(soft_hits_15):.2f}")

print(f"recall@5: {sum(recall_5)/len(recall_5):.2f}")
print(f"mrr: {sum(mrr)/len(mrr):.2f}")