import os
import pyarrow.parquet as pq
import pyarrow as pa

def split_parquet_file():
    input_file = input("Enter the input Parquet file name (including path if not in current directory): ").strip()
    output_folder_name = input("Enter the output folder name: ").strip()

    base_output_path = "/storage/ammar-temp"
    output_path = os.path.join(base_output_path, output_folder_name)

    try:
        os.makedirs(output_path, exist_ok=True)
        print(f"Output directory created at: {output_path}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return

    try:
        table = pq.read_table(input_file)
        total_rows = table.num_rows
        print(f"Total rows in input file: {total_rows}")
    except Exception as e:
        print(f"Error reading input Parquet file: {e}")
        return

    rows_per_file = 10000
    num_parts = (total_rows + rows_per_file - 1) // rows_per_file

    print(f"Splitting into {num_parts} part(s) with up to {rows_per_file} rows each.")

    info_lines = []

    for i in range(num_parts):
        start = i * rows_per_file
        end = min((i + 1) * rows_per_file, total_rows)
        part_table = table.slice(start, end - start)

        part_filename = f"part{i + 1}.parquet"
        part_path = os.path.join(output_path, part_filename)

        try:
            pq.write_table(part_table, part_path)
            print(f"Saved {part_filename} with rows {start + 1}-{end}")
        except Exception as e:
            print(f"Error writing {part_filename}: {e}")
            continue

        info_line = f"{start + 1}-{end} in {part_filename}"
        info_lines.append(info_line)

    info_file_path = os.path.join(output_path, "file_info.txt")
    try:
        with open(info_file_path, "w") as f:
            for line in info_lines:
                f.write(line + "\n")
        print(f"Information about split files saved to {info_file_path}")
    except Exception as e:
        print(f"Error writing information file: {e}")

    print("Splitting completed successfully.")