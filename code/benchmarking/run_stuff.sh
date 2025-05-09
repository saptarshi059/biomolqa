#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

for model in "GCN" "SAGE" "GAT"; do
    for i in {1..3}; do
        python gnn_retriever.py --graph_type $model --run_number $i
    done
done