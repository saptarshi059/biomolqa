#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python gnn_retriever.py --graph_type "GCN"
python gnn_retriever.py --graph_type "SAGE"
python gnn_retriever.py --graph_type "GAT"