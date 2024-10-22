curl -ks <inference_endpoint_url>/v2/models/<model_name>/infer -d '{ "model_name": "<model_name>", "inputs": [{ "name": "<name_of_model_input>", "shape": [<shape>], "datatype": "<data_type>", "data": [<data>] }]}' -H 'Authorization: Bearer <token>'


curl -X POST https://gte-large-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/gte-large/infer    -H "Content-Type: application/json"    -d '{
         "inputs": [
           {
             "name": "attention_mask",
             "shape": [1, 128],
             "datatype": "INT64",
             "data": [1, 1, 1]  # Replace with your actual attention mask data
           },
           {
             "name": "input_ids",
             "shape": [1, 128],
             "datatype": "INT64",
             "data": [101, 2054, 2003]  # Replace with your actual input_ids data
           },
           {
             "name": "token_type_ids",
             "shape": [1, 128],
             "datatype": "INT64",
             "data": [0, 0, 0]  # Replace with your actual token_type_ids data
           }
         ]
       }'