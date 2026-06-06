import os
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets

# 1. Configuration
dataset_name = "roneneldan/TinyStories"
out_dir = "data/processed"
os.makedirs(out_dir, exist_ok=True)

# 2. Initialize Tokenizer
enc = tiktoken.get_encoding("gpt2")

# 3. Load Dataset
import datasets
datasets.enable_progress_bar()
datasets.logging.set_verbosity_info()
print(f"Loading dataset {dataset_name} (this downloads ~1GB of files, please wait)...")
dataset = load_dataset(dataset_name, split="train")

def process(example):
    ids = enc.encode_ordinary(example['text']) # encode text
    ids.append(enc.eot_token) # add end-of-text token
    return {'ids': ids, 'len': len(ids)}

# 4. Map & Tokenize (Use all CPU cores!)
tokenized = dataset.map(
    process,
    remove_columns=['text'],
    desc="Tokenizing dataset",
    num_proc=8, # Matches typical cores, adjust if needed
)

# Shuffle the dataset before saving to prevent learning bias
print("Shuffling the dataset...")
tokenized = tokenized.shuffle(seed=42)

# 5. Save as Binary (Numpy memmap style)
print("Saving to binary memmap format...")

# Summing the 'len' column is fast and RAM-friendly
print("Calculating total token count...")
total_tokens = sum(tokenized['len'])
print(f"Total tokens: {total_tokens:,}")

filename = os.path.join(out_dir, 'train.bin')
train_bin = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(total_tokens,))

# Write in slices to keep memory usage low
write_ptr = 0
total_examples = len(tokenized)
batch_size = 2048

print("Writing tokens to memmap...")
for step in range(0, total_examples, batch_size):
    batch = tokenized[step : step + batch_size]
    # Flatten the batch of ids
    flat_ids = [idx for ids in batch['ids'] for idx in ids]
    arr = np.array(flat_ids, dtype=np.uint16)
    
    # Write to memmap
    train_bin[write_ptr : write_ptr + len(arr)] = arr
    write_ptr += len(arr)
    
    if step % 20480 == 0 or (step + batch_size) >= total_examples:
        print(f"Progress: {write_ptr:,} / {total_tokens:,} tokens written ({(write_ptr / total_tokens * 100):.1f}%)")

train_bin.flush()
print(f"Total tokens saved: {write_ptr:,}")
