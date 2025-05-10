import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch.optim import Adam

# Set random seed for reproducibility
torch.manual_seed(42)

# Function to load triples and create a graph
def load_triples_and_create_graph(triples_file):
    """
    Load triples (subject, relation, object) from a file and create a PyTorch Geometric graph.
    Assumes the file format is: subject relation object (space separated)
    """
    # Read triples
    df = pd.read_csv("../../data/mined_data/full_graph.csv")
    triples = []
    for row in df.itertuples():
        triples.append((row.entity_1, row.relationship, row.entity_2, row.label))
    
    # Create entity and relation dictionaries
    entities = sorted(list(set([t[0] for t in triples] + [t[2] for t in triples])))
    relations = sorted(list(set([t[1] for t in triples])))
    
    entity_to_idx = {ent: idx for idx, ent in enumerate(entities)}
    relation_to_idx = {rel: idx for idx, rel in enumerate(relations)}
    
    # Create edge index and edge type
    edge_index = []
    edge_type = []
    
    for s, r, o, l in triples:
        # Add forward edge
        edge_index.append([entity_to_idx[s], entity_to_idx[o]])
        edge_type.append(relation_to_idx[r])
        
        if l == 0:
            # Add backward edge (for undirected graph representation)
            edge_index.append([entity_to_idx[o], entity_to_idx[s]])
            edge_type.append(relation_to_idx[r] + len(relations))  # Different ID for backward relation
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    
    # Create random initial node features if none available
    num_entities = len(entities)
    num_features = 64  # You can adjust this
    x = torch.randn(num_entities, num_features)
    
    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    
    return data, entity_to_idx, relation_to_idx

# Self-Supervised GNN Model with link prediction objective
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

# Function to generate negative samples
def generate_negative_samples(edge_index, num_nodes, num_neg_samples):
    """Generate negative samples (edges that don't exist in the graph)"""
    # Convert edge index to set for O(1) lookup
    edge_set = set([(i.item(), j.item()) for i, j in edge_index.t()])
    
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        if i != j and (i, j) not in edge_set and (j, i) not in edge_set:
            neg_edges.append([i, j])
            edge_set.add((i, j))  # Add to set to avoid duplicates
    
    return torch.tensor(neg_edges, dtype=torch.long).t()

# Link prediction training with self-supervised learning
def train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1):
    """Split edges into train/val/test sets for link prediction"""
    num_nodes = data.x.size(0)
    row, col = data.edge_index
    
    # Filter out self-loops
    mask = row != col
    row, col = row[mask], col[mask]
    
    n_edges = row.size(0)
    
    # Create positive edges for train/val/test with random split
    all_edges = torch.stack([row, col], dim=0)
    all_indices = torch.randperm(n_edges)
    
    test_size = int(n_edges * test_ratio)
    val_size = int(n_edges * val_ratio)
    train_size = n_edges - test_size - val_size
    
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]
    
    train_pos_edge_index = all_edges[:, train_indices]
    val_pos_edge_index = all_edges[:, val_indices]
    test_pos_edge_index = all_edges[:, test_indices]
    
    # Create same number of negative samples for validation and test
    val_neg_edge_index = generate_negative_samples(all_edges, num_nodes, val_size)
    test_neg_edge_index = generate_negative_samples(all_edges, num_nodes, test_size)
    
    # Create new data object with only training edges
    data.train_pos_edge_index = train_pos_edge_index
    data.val_pos_edge_index = val_pos_edge_index
    data.val_neg_edge_index = val_neg_edge_index
    data.test_pos_edge_index = test_pos_edge_index
    data.test_neg_edge_index = test_neg_edge_index
    
    return data

# Training function
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    
    # Generate negative samples for training
    neg_edge_index = generate_negative_samples(
        data.train_pos_edge_index, data.x.size(0), data.train_pos_edge_index.size(1))
    
    # Forward pass
    z = model.encode(data.x, data.train_pos_edge_index)
    
    # Positive samples
    pos_out = model.decode(z, data.train_pos_edge_index)
    
    # Negative samples
    neg_out = model.decode(z, neg_edge_index)
    
    # Binary cross entropy loss
    pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))
    
    loss = pos_loss + neg_loss
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Evaluation function
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)
        
        # Validation
        pos_val_out = model.decode(z, data.val_pos_edge_index)
        neg_val_out = model.decode(z, data.val_neg_edge_index)
        
        # Test
        pos_test_out = model.decode(z, data.test_pos_edge_index)
        neg_test_out = model.decode(z, data.test_neg_edge_index)
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        
        val_pred = torch.cat([pos_val_out, neg_val_out]).cpu().numpy()
        val_true = torch.cat([torch.ones_like(pos_val_out), torch.zeros_like(neg_val_out)]).cpu().numpy()
        val_auc = roc_auc_score(val_true, val_pred)
        
        test_pred = torch.cat([pos_test_out, neg_test_out]).cpu().numpy()
        test_true = torch.cat([torch.ones_like(pos_test_out), torch.zeros_like(neg_test_out)]).cpu().numpy()
        test_auc = roc_auc_score(test_true, test_pred)
        
    return val_auc, test_auc

# Main function to run the code
def main(triples_file='graph_triples.txt', epochs=200):
    # Load data
    print("Loading triples and creating graph...")
    data, entity_to_idx, relation_to_idx = load_triples_and_create_graph(triples_file)
    
    # Split edges for link prediction
    print("Splitting edges for link prediction task...")
    data = train_test_split_edges(data)
    
    # Create model
    print("Creating model...")
    in_channels = data.x.size(1)
    model = SelfSupervisedGNN(in_channels=in_channels, hidden_channels=128, out_channels=64)
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # Train model
    print("Training model...")
    best_val_auc = 0
    best_test_auc = 0
    
    for epoch in range(1, epochs + 1):
        loss = train(model, optimizer, data)
        val_auc, test_auc = evaluate(model, data)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc = test_auc
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Test AUC: {test_auc:.4f}')
    
    print(f"Training complete. Best Test AUC: {best_test_auc:.4f}")
    
    # Get final node embeddings
    final_embeddings = model.encode(data.x, data.train_pos_edge_index).detach()
    
    # Create entity to embedding mapping
    idx_to_entity = {idx: ent for ent, idx in entity_to_idx.items()}
    entity_embeddings = {idx_to_entity[i]: emb.numpy() for i, emb in enumerate(final_embeddings)}
    
    print(f"Final embeddings shape: {final_embeddings.shape}")
    return model, entity_embeddings

# Example usage
if __name__ == "__main__":
    model, embeddings = main('../../data/mined_data/full_graph.csv')
    torch.save(model, "saved_models/graph_trained.pt")
    torch.save(embeddings, "saved_models/node_embeddings.pt")