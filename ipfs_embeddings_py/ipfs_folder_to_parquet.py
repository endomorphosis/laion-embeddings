import os
import pandas as pd
from datasets import Dataset
import pyarrow.parquet as pq
import tempfile
class ipfs_folder_to_parquet:
    def __init__(self, resources, metadata):
        self.resources = resources
        self.metadata = metadata

    def main(self, input_path, output_path):
        if self.is_parquet(input_path) and self.is_folder(output_path):
            self.parquet_to_folder(input_path, output_path)
        elif self.is_folder(input_path) and self.is_parquet(output_path):
            self.folder_to_parquet(input_path, output_path)
        else:
            raise ValueError("Unable to determine conversion direction.")

    def is_parquet(self, path):
        return os.path.isfile(path) and path.endswith('.parquet')

    def is_folder(self, path):
        return os.path.isdir(path)

    def folder_to_parquet(self, folder_path, parquet_path):
        files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]
        data = []
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'rb') as f:
                content = f.read()
            data.append({'file_name': file_name, 'content': content})
        df = pd.DataFrame(data)
        pq.write_table(df, parquet_path)

    def folder_to_huggingface_dataset(self, folder_path):
        files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
        ]
        data = []
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'rb') as f:
                content = f.read()
            data.append({'file_name': file_name, 'content': content})
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        return dataset
    
    def parquet_to_folder(self, parquet_path, folder_path):
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        os.makedirs(folder_path, exist_ok=True)
        for _, row in df.iterrows():
            file_path = os.path.join(folder_path, row['file_name'])
            with open(file_path, 'wb') as f:
                f.write(row['content'])