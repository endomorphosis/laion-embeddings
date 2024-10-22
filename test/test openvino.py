# import numpy as np
# from ovmsclient import make_grpc_client, make_http_client
from transformers import BertTokenizer
import os 
import requests
import subprocess

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
# Sample text
text = "What is the capital of France?"

# Tokenize
encoded_input = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

# Display
print("Input IDs:", encoded_input['input_ids'])
print("Attention Mask:", encoded_input['attention_mask'])
print("Token Type IDs:", encoded_input['token_type_ids'])

input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']
token_type_ids = encoded_input['token_type_ids']

curl = """ curl -X POST https://gte-large-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/gte-large/infer    -H "Content-Type: application/json"    -d '{
         "inputs": [
           {
             "name": "attention_mask",
             "shape": [1, 128],
             "datatype": "INT64",
             "data": {attention_mask}  # Replace with your actual attention mask data
           },
           {
             "name": "input_ids",
             "shape": [1, 128],
             "datatype": "INT64",
             "data": {input_ids}  # Replace with your actual input_ids data
           },
           {
             "name": "token_type_ids",
             "shape": [1, 128],
             "datatype": "INT64",
             "data": {token_type_ids}  # Replace with your actual token_type_ids data
           }
         ]
       }'
       """

results = subprocess.check_output(curl)
print(results)
