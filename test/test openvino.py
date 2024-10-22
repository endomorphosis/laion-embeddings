from transformers import BertTokenizer
import os 
import requests
import subprocess

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
# Sample text
text = "What is the capital of France?"


class ModelID:
    def __init__(self, value):
        self.value = value

model_id = ModelID("Alibaba-NLP/gte-Qwen2-1.5B-instruct")


# Tokenize
encoded_input = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')

# Display
# print("Input IDs:", encoded_input['input_ids'])
# print("Attention Mask:", encoded_input['attention_mask'])
# print("Token Type IDs:", encoded_input['token_type_ids'])

input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']
token_type_ids = encoded_input['token_type_ids']

# Request
url = "https://gte-large-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/gte-large/infer"
headers = {"Content-Type": "application/json"}
data = {
    "inputs": [
        {
            "name": "attention_mask",
            "shape": [1, 128],
            "datatype": "INT64",
            "data": attention_mask.tolist()
        },
        {
            "name": "input_ids",
            "shape": [1, 128],
            "datatype": "INT64",
            "data": input_ids.tolist()
        },
        {
            "name": "token_type_ids",
            "shape": [1, 128],
            "datatype": "INT64",
            "data": token_type_ids.tolist()
        }
    ]
}
response = requests.post(url, headers=headers, json=data)
results = response.json()
print(results)



