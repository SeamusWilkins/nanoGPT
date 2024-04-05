import os
import re
import numpy as np
import tiktoken

# Path to your custom dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'data.txt')

# Read the dataset
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# Function to preprocess the data
def preprocess_data(text):
    # Remove lines with "Media omitted"
    text = re.sub(r'.*<Media omitted>\n?', '', text)
    # Remove timestamps and sender names
    text = re.sub(r'\d{1,2}\/\d{1,2}\/\d{4}, \d{2}:\d{2} - .*?: ', '', text)
    # Optionally, add more preprocessing steps here
    return text

# Apply preprocessing
data = preprocess_data(data)

# Encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_data = data[:int(len(data)*0.9)]
val_data = data[int(len(data)*0.9):]
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))