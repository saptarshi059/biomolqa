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

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", default=1e-4,type=float)
parser.add_argument("--graph_type", default="GCNConv", type=str)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--heads",default=1,type=int)
parser.add_argument("--run_number", default=1, type=int)
parser.add_argument("--query_embedding_model", default='sentence-transformers/all-MiniLM-L6-v2', type=str)
args = parser.parse_args()

df = pd.read_csv("../../data/mined_data/full_graph.csv")

# --- Create mapping from entities and relations to IDs ---
entity2id = defaultdict(lambda: len(entity2id))
relation2id = defaultdict(lambda: len(relation2id))

edges = []
edge_types = []

# --- Process each row in the dataframe to build edge list ---
for row in df.itertuples():
    h = entity2id[row.entity_1]
    t = entity2id[row.entity_2]
    r = relation2id[row.relationship]

    edges.append((h, t))
    edge_types.append(r)

    # Optional: make the edge bidirectional if label == 0
    if row.label == 0:
        edges.append((t, h))
        edge_types.append(r)

# --- Convert to PyTorch tensors ---
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Shape: [2, num_edges]
edge_attr = torch.tensor(edge_types, dtype=torch.long)              # Shape: [num_edges]

# --- Node features: learnable embeddings ---
num_nodes = len(entity2id)
embedding_dim = 64  # You can change this to any embedding dimension you prefer
embedding = nn.Embedding(num_nodes, embedding_dim)
x = embedding.weight  # Shape: [num_nodes, embedding_dim]

# --- Create the PyTorch Geometric Data object ---
graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

"""# GNN Train"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        x = self.relu(self.linear(self.conv2(x, edge_index)))
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

def custom_collate_fn_train(batch):
    queries, positive_triple, negative_triple = zip(*batch)
    queries = torch.stack(queries)
    return queries, positive_triple, negative_triple

def custom_collate_fn_test(batch):
    queries, positive_triple = zip(*batch)
    queries = torch.stack(queries)
    return queries, positive_triple

def create_triple_embeddings(triple_list, node_embeddings):
  embs = []
  for triple in triple_list:
    embs.append(triple_encoder(entity2id[triple[0]], relation2id[triple[1]], entity2id[triple[2]], node_embeddings))
  return torch.stack(embs).reshape(len(triple_list), 1, query_encoder.bert.config.hidden_size)

def hard_hits(preds, gold):
  s = set()
  for x in gold:
    s.add(tuple(x.tolist()))
  return int(set(s).issubset(set(preds)))

def soft_hits(preds, gold):
    gold = [tuple(x.tolist()) for x in gold]
    for item in preds:
      if item in gold:
        return 1
    return 0

def recall(gold_list, retrieved_list):
    s = set()
    for x in gold_list:
        s.add(tuple(x.tolist()))
    return len(set(s).intersection(set(retrieved_list))) / len(s) if s else 0

def mrr_calc(gold_list, retrieved_list):
    gold_list = [tuple(x.tolist()) for x in gold_list]
    for idx, item in enumerate(retrieved_list):
        if item in gold_list:
            return 1/(idx+1)
    return 0

def test_samples():
    gcn.eval()
    query_encoder.eval()
    triple_encoder.eval()

    all_head_ids = torch.tensor([entity2id[h] for h in df["entity_1"]], dtype=torch.long)
    all_rel_ids = torch.tensor([relation2id[r] for r in df["relationship"]], dtype=torch.long)
    all_tail_ids = torch.tensor([entity2id[t] for t in df["entity_2"]], dtype=torch.long)

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

    for batch in tqdm(test_dataloader):
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
            predictions = []
            for idx, index_list in enumerate(top5_indices):
              for index in index_list:
                predictions.append((H[index], R[index], T[index]))
              hard_hits_5.append(hard_hits(predictions, batch[1][idx]))
              soft_hits_5.append(soft_hits(predictions, batch[1][idx]))
              recall_5.append(recall(batch[1][idx], predictions))
              mrr.append(mrr_calc(batch[1][idx], predictions))

            top10 = torch.topk(sims, k=10)
            top10_indices = top10.indices.tolist()
            for idx, index_list in enumerate(top10_indices):
              for index in index_list:
                predictions.append((H[index], R[index], T[index]))
              hard_hits_10.append(hard_hits(predictions, batch[1][idx]))
              soft_hits_10.append(soft_hits(predictions, batch[1][idx]))

            top15 = torch.topk(sims, k=15)
            top15_indices = top15.indices.tolist()
            for idx, index_list in enumerate(top15_indices):
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

query_encoder = QueryEncoder()
gcn = GCN(in_channels=graph.x.size(1), hidden_channels=64, out_channels=128, vector_emb_dim=query_encoder.bert.config.hidden_size, graph_type=args.graph_type)
triple_encoder = TripleEmbedder(node_embed_dim=query_encoder.bert.config.hidden_size, num_rels=graph.num_edges)

train_df = pd.read_parquet("../../data/mined_data/train_gold.parquet")
train_dataset = TrainGraphDataset(train_df)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn_train, worker_init_fn=seed_worker, generator=g)

test_df = pd.read_parquet("../../data/mined_data/test_gold.parquet")
test_dataset = TestGraphDataset(test_df)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn_test)

optimizer = torch.optim.AdamW(
    list(gcn.parameters()) +
    list(query_encoder.parameters()) +
    list(triple_encoder.parameters()), lr=args.learning_rate)

triplet_loss = nn.TripletMarginLoss()

gcn.train()
query_encoder.train()
triple_encoder.train()

for epoch in tqdm(range(args.epochs)):
    loss_val = 0
    for batch in tqdm(train_dataloader):
      optimizer.zero_grad()

      node_embeddings = gcn(graph)
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
torch.save(gcn, f"saved_models/{args.graph_type}/run_number_{args.run_number}/gcn.pt")
torch.save(gcn, f"saved_models/{args.graph_type}/run_number_{args.run_number}/query_encoder.pt")
torch.save(gcn, f"saved_models/{args.graph_type}/run_number_{args.run_number}/triple_encoder.pt")