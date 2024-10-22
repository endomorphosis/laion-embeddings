import numpy as np
from ovmsclient import make_grpc_client, make_http_client

client = make_http_client("https://gte-large-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com")

model_metadata = client.get_model_metadata(model_name="gte-large")

print(model_metadata)