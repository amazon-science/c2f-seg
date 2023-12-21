#!/bin/bash

# Check if the argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

# Directory names
dataset="$1"
dirs=( "../check_points/${dataset}_c2f_seg" "../check_points/${dataset}_vqgan" )

# URLs
urls=( "https://data.dgl.ai/dataset/C2F-Seg/vqgan_${dataset}.pth" 
       "https://data.dgl.ai/dataset/C2F-Seg/vqgan_${dataset}.yml" 
       "https://data.dgl.ai/dataset/C2F-Seg/c2f_seg_${dataset}.pth" 
       "https://data.dgl.ai/dataset/C2F-Seg/c2f_seg_${dataset}.yml" )

# Create directories if they don't exist
for dir in "${dirs[@]}"; do
    mkdir -p "$dir"
done

# Download files
index=0
for url in "${urls[@]}"; do
    wget "$url" -P "${dirs[$index]}"
    index=$((index + 1))
done
