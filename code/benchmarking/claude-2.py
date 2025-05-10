# Create Graph First

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
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=1e-4,type=float)
parser.add_argument("--graph_type", default="GCN", type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--heads",default=1,type=int)
parser.add_argument("--run_number", default=1, type=int)
parser.add_argument("--query_embedding_model", default='sentence-transformers/all-MiniLM-L6-v2', type=str)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelfSupervisedGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def encode(self, x, edge_index):
        # Get node embeddings
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        # Simple dot product decoder for link prediction
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=1)
    
    def decode_all(self, z):
        # Compute pairwise dot products for all possible node pairs
        prob_adj = z @ z.t()
        return prob_adj

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

class TrainGraphDataset(Dataset):
    def __init__(self, df):
        self.questions = df.Question.tolist()
        self.gold_triples = df.Gold_Triples.tolist()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question_emb = query_encoder(self.questions[idx])
        positive_triple = random.choice(self.gold_triples[idx]).tolist()
        while True:
          negative_triple = random.choice(train_df.Gold_Triples.to_list())[0].tolist()
          if negative_triple != positive_triple:
            return question_emb, positive_triple, negative_triple

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

def custom_collate_fn_train(batch):
    queries, positive_triple, negative_triple = zip(*batch)
    queries = torch.stack(queries)
    return queries, positive_triple, negative_triple

def custom_collate_fn_validation(batch):
    queries, positive_triple = zip(*batch)
    queries = torch.stack(queries)
    return queries, positive_triple

def create_triple_embeddings(triple_list, node_embeddings):
  embs = []
  for triple in triple_list:
    embs.append(triple_encoder(entity2id[triple[0]], relation2id[triple[1]], entity2id[triple[2]], node_embeddings))
  return torch.stack(embs).reshape(len(triple_list), 1, query_encoder.bert.config.hidden_size)

def hard_hits(preds, gold):
  hits = 1
  for x in gold:
    if not tuple(x.tolist()) in preds:
      return 0
  return hits

def soft_hits(preds, gold):
  hits = 0
  for x in gold:
    if tuple(x.tolist()) in preds:
      return 1
  return hits

def recall(preds, gold):
    gold = [tuple(x.tolist()) for x in gold]
    matches = 0
    for x in preds:
      if x in gold:
        matches+=1
    return matches/len(preds)

def mrr_calc(preds, gold):
    gold = [tuple(x.tolist()) for x in gold]
    for idx, item in enumerate(preds):
        if item in gold:
            return 1/(idx+1)
    return 0

def test_samples():
    gcn.eval()
    query_encoder.eval()
    triple_encoder.eval()

    H = df["entity_1"].tolist()
    R = df["relationship"].tolist()
    T = df["entity_2"].tolist()

    hard_hits_5 = []
    soft_hits_5 = []
    
    hard_hits_10 = []
    soft_hits_10 = []
    
    hard_hits_15 = []
    soft_hits_15 = []
    
    recall_5 = []
    mrr = []

    for batch in tqdm(validation_dataloader):
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
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

torch.manual_seed(0)
g = torch.Generator()
g.manual_seed(0)

df = pd.read_csv("../../data/mined_data/full_graph.csv")

graph = torch.load("saved_models/graph_data.pt", weights_only=False)

with Path("saved_models/entity2id.pkl").open("rb") as file:
    entity2id = pickle.load(file)

with Path("saved_models/relation2id.pkl").open("rb") as file:
    relation2id = pickle.load(file)

with Path("saved_models/node_embeddings.pkl").open("rb") as file:
    node_embeddings = pickle.load(file)

gcn = torch.load("saved_models/graph_trained.pt", weights_only=False)

if hasattr(gcn, 'embedding') and isinstance(gcn.embedding, torch.nn.Module):
    with torch.no_grad():
        if hasattr(gcn.embedding, 'weight'):
            # Make sure dimensions match
            assert gcn.embedding.weight.shape == node_embeddings.shape, "Embedding dimensions don't match"
            gcn.embedding.weight.copy_(node_embeddings)
        # If your model uses a different attribute name for embeddings
        elif hasattr(gcn.embedding, 'embedding'):
            assert gcn.embedding.embedding.shape == node_embeddings.shape, "Embedding dimensions don't match"
            gcn.embedding.embedding.copy_(node_embeddings)

query_encoder = QueryEncoder()
#gcn = GCN(in_channels=graph.x.size(1), hidden_channels=32, out_channels=32, vector_emb_dim=query_encoder.bert.config.hidden_size, graph_type=args.graph_type)
triple_encoder = TripleEmbedder(node_embed_dim=query_encoder.bert.config.hidden_size, num_rels=graph.num_edges)

train_df = pd.read_parquet("../../data/mined_data/train_gold.parquet")
train_dataset = TrainGraphDataset(train_df)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn_train, worker_init_fn=seed_worker, generator=g)

validation_df = pd.read_parquet("../../data/mined_data/validation_gold.parquet")
validation_dataset = ValidationGraphDataset(validation_df)
validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn_validation)

all_head_ids = torch.tensor([entity2id[h] for h in df["entity_1"]], dtype=torch.long)
all_rel_ids = torch.tensor([relation2id[r] for r in df["relationship"]], dtype=torch.long)
all_tail_ids = torch.tensor([entity2id[t] for t in df["entity_2"]], dtype=torch.long)

optimizer = torch.optim.AdamW(list(gcn.parameters()) + list(query_encoder.parameters()) + list(triple_encoder.parameters()), lr=args.learning_rate)
triplet_loss = nn.TripletMarginLoss()

for epoch in tqdm(range(args.epochs)):
    gcn.train()
    query_encoder.train()
    triple_encoder.train()

    loss_val = 0
    for batch in tqdm(train_dataloader):
      optimizer.zero_grad()
    
      node_embeddings = gcn.encode(graph.x, graph.edge_index)
      query_embeddings = batch[0]  # [B, D]

      positive_triple_embeddings = create_triple_embeddings(batch[1], node_embeddings)
      negative_triple_embeddings = create_triple_embeddings(batch[2], node_embeddings)

      loss = triplet_loss(query_embeddings, positive_triple_embeddings, negative_triple_embeddings)
      loss_val += loss.item()

      loss.backward()
      optimizer.step()

    epoch_loss = loss_val / len(train_dataloader)
    print(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

    test_samples()

print("Saving models...")
savepath = Path(f"saved_models/{args.graph_type}/run_number_{args.run_number}/")
savepath.mkdir(parents=True, exist_ok=True)

torch.save(gcn, savepath / "gcn.pt")
torch.save(query_encoder, savepath / "query_encoder.pt")
torch.save(triple_encoder, savepath / "triple_encoder.pt")