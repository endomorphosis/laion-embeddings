import os
import pyarrow.parquet as pq
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import shutil

def flatten_embeddings(embeddings):
    return [item for sublist in embeddings for item in sublist]

def cluster_parquet_splits():
    split_folder_name = input("Enter the split folder name (under /storage/ammar-temp): ").strip()
    base_output_path = "/storage/ammar-temp"
    split_path = os.path.join(base_output_path, split_folder_name)
    
    if not os.path.isdir(split_path):
        print(f"The directory {split_path} does not exist.")
        return
    
    split_files = [f for f in os.listdir(split_path) if f.endswith('.parquet')]
    
    if not split_files:
        print(f"No Parquet files found in {split_path}.")
        return
    
    print(f"Found {len(split_files)} Parquet split files. Processing...")
    
    representative_embeddings = []
    file_names = []
    
    for file in split_files:
        file_path = os.path.join(split_path, file)
        try:
            table = pq.read_table(file_path, columns=["Embeddings"])
            embeddings = table.column("Embeddings").to_pandas().tolist()
            
            if not embeddings:
                print(f"No embeddings found in {file}. Skipping.")
                continue
            
            flat_embeddings = [flatten_embeddings(embed) for embed in embeddings if embed is not None]
            if not flat_embeddings:
                print(f"No valid embeddings found in {file}. Skipping.")
                continue
            
            embeddings_np = np.array(flat_embeddings)
            
            if embeddings_np.ndim == 1:
                embeddings_np = np.stack(embeddings_np)
            
            mean_embedding = np.mean(embeddings_np, axis=0)
            representative_embeddings.append(mean_embedding)
            file_names.append(file)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    if not representative_embeddings:
        print("No valid embeddings found in any split files.")
        return
    
    embeddings_matrix = np.vstack(representative_embeddings)
    print(f"Computed representative embeddings for {embeddings_matrix.shape[0]} files.")
    
    range_min = 2
    range_max = min(10, embeddings_matrix.shape[0] - 1) if embeddings_matrix.shape[0] > 1 else 1
    best_k = 2
    best_score = -1
    
    for k in range(range_min, range_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings_matrix)
        score = silhouette_score(embeddings_matrix, labels)
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"Optimal number of clusters determined: {best_k} with a silhouette score of {best_score:.4f}")
    
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(embeddings_matrix)
    
    clustering_result = {file_names[i]: int(labels[i]) for i in range(len(file_names))}
    
    clustering_file_path = os.path.join(split_path, "clustering_results.json")
    try:
        with open(clustering_file_path, "w") as f:
            json.dump(clustering_result, f, indent=4)
        print(f"Clustering results saved to {clustering_file_path}")
    except Exception as e:
        print(f"Error saving clustering results: {e}")
    
    organize = input("Do you want to organize split files into cluster-specific folders in a new output folder? (y/n): ").strip().lower()
    if organize == 'y':
        new_cluster_folder = input("Enter the name for the new output folder (under /storage/ammar-temp): ").strip()
        new_cluster_path = os.path.join(base_output_path, new_cluster_folder)
        try:
            os.makedirs(new_cluster_path, exist_ok=True)
            print(f"New output folder created at: {new_cluster_path}")
        except Exception as e:
            print(f"Error creating new output folder: {e}")
            return
        for file, cluster in clustering_result.items():
            cluster_dir = os.path.join(new_cluster_path, f"cluster_{cluster}")
            os.makedirs(cluster_dir, exist_ok=True)
            src_path = os.path.join(split_path, file)
            dst_path = os.path.join(cluster_dir, file)
            try:
                shutil.copy(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {file} to cluster {cluster}: {e}")
        print("Files have been organized into cluster-specific folders in the new output folder.")
    
    print("Clustering process completed successfully.")

if __name__ == "__main__":
    cluster_parquet_splits()