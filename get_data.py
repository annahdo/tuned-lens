import json
import random
import os
import pyzstd

# Set a seed for reproducibility

# download dataset
os.system("wget -q -O val.jsonl.zst https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst")


# unzip dataset
with open('val.jsonl.zst', 'rb') as compressed_file:
    compressed_data = compressed_file.read()

decompressed_data = pyzstd.decompress(compressed_data)

with open('val.jsonl', 'wb') as decompressed_file:
    decompressed_file.write(decompressed_data)


random.seed(42)

# Load the original dataset
with open('val.jsonl', 'r') as file:
    original_dataset = [json.loads(line) for line in file]

# Calculate the number of samples for each subset
total_samples = len(original_dataset)


val_size = 0.8*total_samples  
test_size = 0.2*total_samples 

# Randomly shuffle the dataset
random.shuffle(original_dataset)

# Split the dataset into validation and test subsets
val_subset = original_dataset[:val_size]
test_subset = original_dataset[val_size:val_size+test_size]

# Write the subsets to new files
with open('val_subset.jsonl', 'w') as val_file:
    for sample in val_subset:
        val_file.write(json.dumps(sample) + '\n')

with open('test_subset.jsonl', 'w') as test_file:
    for sample in test_subset:
        test_file.write(json.dumps(sample) + '\n')
