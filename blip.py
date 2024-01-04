import torch
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import pdb

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
img_pth = '/home/ubuntu/scratch/tongchen/ctSPML/data/coco/val2014/COCO_val2014_000000000073.jpg'
raw_image = Image.open(img_pth).convert('RGB')

pdb.set_trace()
question = "Look at the entire image carefully and answer the question.\nQuestion: Does a dog appear in the image? Answer:"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda", torch.float16)
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
