import time
import torch
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import pdb

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"

print(fmt.format("start connecting to Milvus"))
connections.connect(alias="default", host="localhost", port="19530")

has = utility.has_collection("label_embedding")
print(f"Does collection label_embedding exist in Milvus: {has}")

print(fmt.format("Start loading"))
collection_name = 'label_embedding'
collection = Collection(name=collection_name)
collection.load()

print(fmt.format("Start searching based on vector similarity"))
x = torch.randn(768).detach().numpy()
search_params = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10},
}

start_time = time.time()
result = collection.search([x], "embeddings", search_params, limit=collection.num_entities)
end_time = time.time()

count = 0
for hits in result:
    for hit in hits:
        if hit.distance < 0.5:
            count += 1
            print(f"hit: {hit}, label field: {hit.entity.get('label')}")
print(search_latency_fmt.format(end_time - start_time))
print(f"count={count}")
