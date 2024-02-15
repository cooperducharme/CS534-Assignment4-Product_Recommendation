from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm

def generate_top_ten_v2(keywords = 'fantasy novel short', semantic = 'pos', filename_list = ['2005','2004','2003','2002','2001','2000']):

    # Load pre-trained BERT model and tokenizer
    MODEL = 'bert-base-uncased'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)

    # Encode keyword and text tokens using BERT
    encoded_keywords = tokenizer(keywords.lower(),padding=True , return_tensors='pt').to(device)
    input_ids = encoded_keywords['input_ids']
    attention_mask = encoded_keywords['attention_mask']
    # encoded_texts = tokenizer(list(df['clean_text']), padding=True, truncation=True, return_tensors='pt')['input_ids']

    # filename_list = ['2005','2004','2003','2002','2001','2000']
    # filename_list = ['2005']
    df_lst = []

    # Generate BERT embeddings for keyword and text tokens batch-wise and compute cosine similarity
    with torch.no_grad():
        keyword_embeddings = model(input_ids.to(device), attention_mask=attention_mask)[0][:, 0, :].cpu().numpy()
    for filename in filename_list:
        # Load DataFrame
        try:
            df = pd.read_pickle('book-pickle-4/'+ filename +'.pkl')
        except(FileNotFoundError):
            continue
        # print(df['keywords_embedding'].at[10])
        # return 0

        df['similarity_score'] = ''


        for idx in tqdm(range(len(df))):

            text_embeddings = df.iloc[idx]['keywords_embedding']
            # Compute cosine similarity between keyword vector and text vectors in the batch
            max_similarity_score = max(cosine_similarity(keyword_embeddings, text_embeddings)[0])
            # Add similarity scores to DataFrame
            df.at[idx, 'similarity_score'] = max_similarity_score
        df_lst.append(df)

    big_df = pd.concat(df_lst)
    
    # filter semantic flag
    if semantic == 'neg':
        big_df = big_df[big_df['sentiment-score'] < -0.3]
    elif semantic == 'pos':
        big_df = big_df[big_df['sentiment-score'] > 0.3]
    print(big_df.head(1))
    # Sort DataFrame by similarity score in descending order and show top 10 rows
    top_10 = big_df.sort_values(by='similarity_score', ascending=False)
    top_10 = top_10.head(10)
    print(top_10[['product_title','clean_text']])
    return top_10[['product_title','clean_text']],"None"



if __name__ == "__main__":
    generate_top_ten_v2()