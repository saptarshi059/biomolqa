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
parser.add_argument("--learning_rate", default=1e-4,type=float)
parser.add_argument("--graph_type", default="GCN", type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--heads",default=1,type=int)
parser.add_argument("--run_number", default=1, type=int)
parser.add_argument("--query_embedding_model", default='sentence-transformers/all-MiniLM-L12-v2', type=str)
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



class ValidationGraphDataset(Dataset):
    def __init__(self, df):
        self.questions = df.Question.tolist()
        self.gold_triples = df.Gold_Triples.tolist()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question_emb = query_encoder(self.questions[idx])
        positive_triples = self.gold_triples[idx].tolist()
        return question_emb, positive_triples


model = torch.load("saved_models/GCN/run_number_1/gcn.pt", weights_only=False)