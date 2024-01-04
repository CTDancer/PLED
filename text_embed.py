import torch
import numpy as np
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from preproc.consts import label_list

model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336",output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14-336")

# text_labels = []
# with open("text_corpus.txt", "r") as f:
#     for sentence in f:
#         text_labels.append(sentence.strip())

text_labels = ["A photo of " + label for label in label_list]
        
# print(len(text_labels))
# print(text_labels[0])

text_labels_token = tokenizer(text_labels, padding=True, return_tensors="pt")
outputs = model(**text_labels_token)
text_labels_embeds = outputs.text_embeds
print(text_labels_embeds.shape) # torch.Size([6801, 768])
np.save('label_embeds.npy', text_labels_embeds.detach().numpy())
# torch.save(text_labels_embeds, 'corpus_embeds.pt')