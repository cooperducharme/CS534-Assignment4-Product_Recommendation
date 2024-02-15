from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load pre-trained BERT model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

MODEL = 'distilbert-base-uncased'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL).to(device)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)

filename_list = ['2003','2002','2001','2000']

for filename in filename_list:
    # Load DataFrame
    try:
        df = pd.read_pickle('book-pickle-3/'+ filename +'.pkl')
    except:
        continue
    # print(df.head(10))
    # break
    # batch_size = 64
    # total_batches = (len(df) + batch_size - 1) // batch_size

    df['keywords_embedding'] = ''
    df['keywords_embedding'] = df['keywords_embedding'].astype(object)
    embeddings_list = []
    # Generate BERT embeddings for keyword and text tokens batch-wise and compute cosine similarity
    with torch.no_grad():
        for idx in tqdm(range(len(df))):

            # Encode texts in the batch
            batch_text_tokens = df.iloc[idx]['keywords']
            try:
                encoded_texts = tokenizer(batch_text_tokens,padding=True, return_tensors='pt').to(device)
            except:
                continue
            # print(encoded_texts)
            # break
            # df.iloc[start_idx:end_idx, df.columns.get_loc('encoded_text')] = encoded_texts
            # Generate BERT embeddings for the batch
            input_ids = encoded_texts['input_ids']
            attention_mask = encoded_texts['attention_mask']
            text_embeddings = model(input_ids.to(device), attention_mask=attention_mask)[0][:, 0, :].cpu().numpy()
            
            # for index in range(len(batch_text_tokens)):
            df.at[idx,'keywords_embedding'] = text_embeddings

    df.to_pickle("book-pickle-4/"+ filename +".pkl")
    
