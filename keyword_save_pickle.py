from keyword_generation import get_keybert_results_with_vectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load pre-trained BERT model and tokenizer
MODEL = 'bert-base-uncased'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)

filename_list = ['2002','2001','2000']

for filename in filename_list:
    # Load DataFrame
    df = pd.read_pickle('book-pickle-2/'+ filename +'.pkl')

    batch_size = 64
    total_batches = (len(df) + batch_size - 1) // batch_size

    df['keywords'] = ''
    df['keywords'] = df['keywords'].astype(object)
    embeddings_list = []
    # Generate BERT embeddings for keyword and text tokens batch-wise and compute cosine similarity
    with torch.no_grad():
        for i in tqdm(range(total_batches)):
            # Get batch start and end indices
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(df))
            
            # Encode texts in the batch
            batch_text_tokens = df.iloc[start_idx:end_idx]['clean_text']
            batch_text_tokens = get_keybert_results_with_vectorizer(batch_text_tokens,device = device)
            # print(len(batch_text_tokens), end_idx - start_idx)

            for index in range(len(batch_text_tokens)):
                df.at[index + start_idx,'keywords'] = batch_text_tokens[index]

    df.to_pickle("book-pickle-3/"+ filename +".pkl")