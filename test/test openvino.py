import numpy as np
from classes import imagenet_classes
from ovmsclient import make_grpc_client


client = make_grpc_client("https://gte-large-endomorphosis-dev.apps.cluster.intel.sandbox1234.opentlc.com/v2/models/gte-large")

with open("zebra.jpeg", "rb") as f:
   img = f.read()

output = client.predict({"0": img}, "resnet")
result_index = np.argmax(output[0])
print(imagenet_classes[result_index])' >> predict.py