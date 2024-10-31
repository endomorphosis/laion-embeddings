from transformers import BertTokenizer
import os 
import requests
import subprocess
import asyncio
import aiohttp
from aiohttp import ClientSession, ClientTimeout
from transformers import AutoTokenizer

async def make_post_request_openvino(endpoint, data):
    headers = {'Content-Type': 'application/json'}
    timeout = ClientTimeout(total=300) 
    async with ClientSession(timeout=timeout) as session:
        try:
            async with session.post(endpoint, headers=headers, json=data) as response:
                if response.status != 200:
                    return ValueError(response)
                return await response.json()
        except Exception as e:
            print(str(e))
            if "Can not write request body" in str(e):
                print( "endpoint " + endpoint + " is not accepting requests")
                return ValueError(e)
            if "Timeout" in str(e):
                print("Timeout error")
                return ValueError(e)
            if "Payload is not completed" in str(e):
                print("Payload is not completed")
                return ValueError(e)
            if "Can not write request body" in str(e):
                return ValueError(e)
            pass
        except aiohttp.ClientPayloadError as e:
            print(f"ClientPayloadError: {str(e)}")
            return ValueError(f"ClientPayloadError: {str(e)}")
        except asyncio.TimeoutError as e:
            print(f"Timeout error: {str(e)}")
            return ValueError(f"Timeout error: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return ValueError(f"Unexpected error: {str(e)}")

# Load the tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = "Qdrant/gte-large-onnx"
model = "Alibaba-NLP/gte-large-en-v1.5"
model = "aapot/bge-m3-onnx"
model = "neoALI/bge-m3-rag-ov"
tokenizer =  AutoTokenizer.from_pretrained(model, device='cpu', use_fast=True)
# Sample text
text = "What is the capital of France?"

# Tokenize
encoded_input = tokenizer(text, max_length=4095, padding='max_length', truncation=True, return_tensors='pt')

# Display
# print("Input IDs:", encoded_input['input_ids'])
# print("Attention Mask:", encoded_input['attention_mask'])
# print("Token Type IDs:", encoded_input['token_type_ids'])

input_ids = encoded_input['input_ids'].tolist()
attention_mask = encoded_input['attention_mask'].tolist()
# token_type_ids = encoded_input['token_type_ids']

# Request
url = "curl https://bge-m3-onnx-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-onnx/infer"
headers = {"Content-Type": "application/json"}
data = {
    "inputs": [
        {
            "name": "attention_mask",
            "shape": [1, 4095],
            "datatype": "INT64",
            "data": attention_mask
        },
        {
            "name": "input_ids",
            "shape": [1, 4095],
            "datatype": "INT64",
            "data": input_ids
        }
        # {
        #     "name": "token_type_ids",
        #     "shape": [1, 128],
        #     "datatype": "INT64",
        #     "data": token_type_ids.tolist()
        # }
    ]
}
# response = requests.post(url, headers=headers, json=data)
# results = response.json()
# print(results)

endpoint = "https://bge-m3-rag-ov-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/bge-m3-rag-ov/infer"
asyncio.run(make_post_request_openvino(endpoint, data))
