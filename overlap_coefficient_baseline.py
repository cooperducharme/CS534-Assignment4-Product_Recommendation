from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
def overlap_coefficient(keyword_list, text_list):
    set1 = set(keyword_list[0])
    set2 = np.apply_along_axis(lambda x: set(x), 1, text_list)

    intersection = np.frompyfunc(lambda x: len(set1.intersection(x)), 1, 1)
    overlap_coefficients = intersection(set2) / keyword_list[0].size

    return overlap_coefficients


def generate_top_ten(keywords = 'fantasy novel short space travel', semantic = 'both', filename_list = ['2005','2004','2003','2002','2001','2000']):

    # Load pre-trained BERT model and tokenizer
    MODEL = 'bert-base-uncased'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)

    # Encode keyword and text tokens using BERT
    encoded_keywords = tokenizer(keywords.lower(), return_tensors='pt')['input_ids']
    # encoded_texts = tokenizer(list(df['clean_text']), padding=True, truncation=True, return_tensors='pt')['input_ids']

    # filename_list = ['2005','2004','2003','2002','2001','2000']
    # filename_list = ['2005']
    df_lst = []

    # Generate BERT embeddings for keyword and text tokens batch-wise and compute cosine similarity
    with torch.no_grad():
        keyword_embeddings = model(encoded_keywords.to(device))[0][:, 0, :].cpu().numpy()
    for filename in filename_list:
        # Load DataFrame
        try:
            df = pd.read_pickle('book-pickle/'+ filename +'.pkl')
        except(FileNotFoundError):
            continue
        # print(df['embeddings'].head(10))
        # break
        batch_size = 64
        total_batches = (len(df) + batch_size - 1) // batch_size

        df['overlap_coefficient'] = ''


        for i in tqdm(range(total_batches)):
            # Get batch start and end indices
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(df))

            text_embeddings = list(df.iloc[start_idx:end_idx]['embeddings'])
            # Compute cosine similarity between keyword vector and text vectors in the batch
            similarity_scores = overlap_coefficient(keyword_embeddings, text_embeddings)
            
            # Add similarity scores to DataFrame
            df.iloc[start_idx:end_idx, df.columns.get_loc('overlap_coefficient')] = similarity_scores
        df_lst.append(df)

    big_df = pd.concat(df_lst)
    
    # filter semantic flag
    if semantic == 'neg':
        big_df = big_df[big_df['sentiment-score'] < -0.3]
    elif semantic == 'pos':
        big_df = big_df[big_df['sentiment-score'] > 0.3]
    
    # Sort DataFrame by similarity score in descending order and show top 10 rows
    top_10 = big_df.sort_values(by='overlap_coefficient', ascending=False)
    top_10 = top_10.head(10)
    print(top_10[['product_title','clean_text','overlap_coefficient']])
    return(top_10[['product_title','clean_text']])



if __name__ == "__main__":
    generate_top_ten()