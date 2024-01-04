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
num_entities, dim = 6801, 768

#################################################################################
# 1. connect to Milvus
# Add a new connection alias `default` for Milvus server in `localhost:19530`
# Actually the "default" alias is a buildin in PyMilvus.
# If the address of Milvus is the same as `localhost:19530`, you can omit all
# parameters and call the method as: `connections.connect()`.
#
# Note: the `using` parameter of the following methods is default to "default".
print(fmt.format("start connecting to Milvus"))
connections.connect(alias="default", host="localhost", port="19530")

has = utility.has_collection("label_embedding")
print(f"Does collection label_embedding exist in Milvus: {has}")

if has is True:
    pdb.set_trace()
    utility.drop_collection("label_embedding")
    exit()

#################################################################################
# 2. create collection
# We're going to create a collection with 3 fields.
# +-+------------+------------+------------------+------------------------------+
# | | field name | field type | other attributes |       field description      |
# +-+------------+------------+------------------+------------------------------+
# |1|    "label"    |   VarChar  |  is_primary=True |      "primary field"         |
# | |            |            |   auto_id=False  |                              |
# +-+------------+------------+------------------+------------------------------+
# |2|"embeddings"| FloatVector|     dim=768        |  "float vector with dim 768"   |
# +-+------------+------------+------------------+------------------------------+
fields = [
    FieldSchema(name="label", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
]

schema = CollectionSchema(fields, "label_embedding stores the embeddings of label texts")

print(fmt.format("Create collection `label_embedding`"))
label_embedding = Collection("label_embedding", schema, consistency_level="Strong")

################################################################################
# 3. insert data
# We are going to insert 3000 rows of data into `label_embedding`
# Data to be inserted must be organized in fields.
#
# The insert() method returns:
# - either automatically generated primary keys by Milvus if auto_id=True in the schema;
# - or the existing primary key field from the entities if auto_id=False in the schema.

print(fmt.format("Start inserting entities"))

label_embeds = torch.load('label_embeds.pt')
labels = []
with open('noun_corpus.txt', 'r') as f:
    for label in f:
        labels.append(label.strip())
        
assert(len(labels) == label_embeds.shape[0])

entities = [
    labels,
    label_embeds.detach().numpy(),    # field embeddings, supports numpy.ndarray and list
]

insert_result = label_embedding.insert(entities)

label_embedding.flush()
print(f"Number of entities in Milvus: {label_embedding.num_entities}")  # check the num_entites

################################################################################
# 4. create index
# We are going to create an IVF_FLAT index for label_embedding collection.
# create_index() can only be applied to `FloatVector` and `BinaryVector` fields.
print(fmt.format("Start Creating index FLAT"))
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128},
}

label_embedding.create_index("embeddings", index)
