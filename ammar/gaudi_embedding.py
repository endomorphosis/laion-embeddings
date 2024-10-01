import pyarrow as pa
import pyarrow.parquet as pq
import requests
import os
import time
from pathlib import Path
import sys

def create_embeddings(input_text, server_url):
    try:
        response = requests.post(server_url, json={"inputs": input_text}, headers={'Content-Type': 'application/json'})
        response.raise_for_status()  # Raise an error for bad status codes
        embedding = response.json()

        if isinstance(embedding, list):
            return embedding
        raise ValueError("Unexpected API response format")
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    except ValueError as ve:
        print(f"Value error: {ve}")
        return None

def process_parquet_file(input_file, output_file, process_all, checkpoint_interval, checkpoint_folder, wiki_language, use_checkpoints, server_url):
    table = pq.read_table(input_file)
    total_rows = table.num_rows
    rows_to_process = min(total_rows, 50) if not process_all else total_rows

    if use_checkpoints:
        os.makedirs(checkpoint_folder, exist_ok=True)

    all_embeddings = []
    all_version_control = []
    all_wiki_language = []
    skipped_rows = []

    for i in range(rows_to_process):
        abstract = table.column('Abstract')[i].as_py()
        version_control = table.column('Version Control')[i].as_py()

        if not abstract:
            skipped_rows.append(i + 1)  # Row numbers start at 1
            print(f"Skipping row {i + 1} due to empty abstract")
            continue

        embeddings = create_embeddings(abstract, server_url)
        if embeddings is None:
            skipped_rows.append(i + 1)
            print(f"Skipping row {i + 1} due to embedding creation failure")
            continue

        all_embeddings.append(embeddings)
        all_version_control.append(version_control)
        all_wiki_language.append(wiki_language)

        print(f"Processing row {i + 1}/{rows_to_process}")

        if use_checkpoints and (i + 1) % checkpoint_interval == 0:
            checkpoint_table = pa.Table.from_pydict({
                'Wiki Language': pa.array(all_wiki_language),
                'Embeddings': pa.array(all_embeddings),
                'Version Control': pa.array(all_version_control)
            })
            checkpoint_file = os.path.join(checkpoint_folder, f'checkpoint_{i + 1}.parquet')
            pq.write_table(checkpoint_table, checkpoint_file)
            print(f"Checkpoint saved at row {i + 1}")
            all_embeddings.clear()
            all_version_control.clear()
            all_wiki_language.clear()

    if all_embeddings:
        final_table = pa.Table.from_pydict({
            'Wiki Language': pa.array(all_wiki_language),
            'Embeddings': pa.array(all_embeddings),
            'Version Control': pa.array(all_version_control)
        })
        pq.write_table(final_table, output_file)
    else:
        combined_tables = []
        if use_checkpoints:
            for file in sorted(os.listdir(checkpoint_folder)):
                if file.endswith(".parquet"):
                    combined_tables.append(pq.read_table(os.path.join(checkpoint_folder, file)))
        if combined_tables:
            combined_table = pa.concat_tables(combined_tables)
            pq.write_table(combined_table, output_file)

    print(f"Final embeddings saved to {output_file}")

    # Print skipped rows summary
    if skipped_rows:
        print("\nSummary of Skipped Rows:")
        print(f"Total skipped rows: {len(skipped_rows)}")
        print("Skipped row numbers:", skipped_rows)
    else:
        print("\nNo rows were skipped.")

if __name__ == "__main__":
    try:
        input_file = input("Enter the input file name (Parquet format): ").strip()
        output_file = input("Enter the output file name (Parquet format): ").strip()

        if not output_file.lower().endswith('.parquet'):
            print("Error: Output file must be a Parquet file (.parquet)")
            sys.exit(1)

        input_file = Path(input_file).resolve()
        output_file = Path(output_file).resolve()

        if not input_file.exists():
            print(f"Error: Input file '{input_file}' does not exist.")
            sys.exit(1)

        wiki_language = input("Enter the Wiki Language (e.g., enwiki, dewiki): ").strip()
        server_url = input("Enter the embedding server URL (e.g., http://159.69.148.218:80/embed): ").strip()

        process_all_input = input("Do you want to process the entire file? (yes/no): ").strip().lower()
        process_all = process_all_input == 'yes'

        use_checkpoints_input = input("Do you want to use checkpoints? (yes/no): ").strip().lower()
        use_checkpoints = use_checkpoints_input == 'yes'

        if use_checkpoints:
            while True:
                try:
                    checkpoint_interval = int(input("Please specify the checkpoint interval (number of rows): ").strip())
                    if checkpoint_interval <= 0:
                        print("Checkpoint interval must be a positive integer.")
                        continue
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid integer for checkpoint interval.")

            checkpoint_folder = input("Please specify the checkpoint folder name: ").strip()
        else:
            checkpoint_interval = None
            checkpoint_folder = None

        start_time = time.time()

        process_parquet_file(
            input_file=input_file,
            output_file=output_file,
            process_all=process_all,
            checkpoint_interval=checkpoint_interval,
            checkpoint_folder=checkpoint_folder,
            wiki_language=wiki_language,
            use_checkpoints=use_checkpoints,
            server_url=server_url
        )

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(f"Processing completed in {elapsed_time:.2f} minutes")

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)