import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm

from functions import get_price_from_list_IDs

# For the Formula : 
HELPFUL_VOTES_MULTIPLIER = 0.001
TOTAL_VOTES_MULTIPLIER = 0.0005
SIMILARITY_MULTIPLIER = 7
SEMANTIC_THRESHOLD = 0.5 ## 0.3, 0.5, 0r 0.7?

ALL_VOTES_MULTIPLIER = 0.0001

FORMULA = 1


def filter_pickle(r_df, search_words):
    ## For a specific product title 
    # df = read_df.loc[read_df['product_title'] == "A Minus Tide" ]

    #First, if there are enough books with the keywords in the title, search for these first and start model with these
    search = (search_words.lower().replace(' ', '|'))
    print("SEARCH  ", search)

    df = r_df[r_df['product_title'].str.contains(search)]
    # If df is not long enough, abandon filter and take all of the books/reviews
    # if len(df) < 5 :
    #     df = r_df

    return df

def generate_top_ten_v3(keywords = 'math science book text', semantic = 'pos', filename_list = ['2005']):

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
    ## Headers: ['product_id', 'product_title', 'review_date', 'star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'clean_text', 'sentiment-score', 'embeddings']

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

        df = filter_pickle(df, keywords)
        # print("List of DF headers ", list(df.columns.values))

        batch_size = 64
        total_batches = (len(df) + batch_size - 1) // batch_size

        df['similarity_score'] = '' 

        for i in tqdm(range(total_batches)):
            # Get batch start and end indices
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(df))

            text_embeddings = list(df.iloc[start_idx:end_idx]['embeddings'])
            # Compute cosine similarity between keyword vector and text vectors in the batch
            similarity_scores = cosine_similarity(keyword_embeddings, text_embeddings)
            
            # Add similarity scores to DataFrame
            df.iloc[start_idx:end_idx, df.columns.get_loc('similarity_score')] = similarity_scores[0].astype(float)
        df_lst.append(df)

    big_df = pd.concat(df_lst)
    # print("   Verified ", big_df['verified_purchase'].head(10))
    # print("   helpful votes ", big_df['helpful_votes'].head(10))
    # print("   total votes ", big_df['total_votes'].head(10))
    # print("   sentiment ", big_df['sentiment-score'].head(10))
    # print("   similarity ", big_df['similarity_score'].head(10))

    # Make new copy dataframe for manipulation
    recommended = big_df[['product_id','product_title','helpful_votes', 'total_votes', 'sentiment-score']].copy()
    # recommended = big_df[['product_id','product_title','star_rating','helpful_votes', 'total_votes', 'sentiment-score']].copy()

    recommended['similarity_score'] =  big_df['similarity_score'].astype('float64')

    # print("  INFO \n", recommended.info())
    
    # filter semantic flag
    if semantic == 'neg':
        recommended = recommended[recommended['sentiment-score'] < -SEMANTIC_THRESHOLD]
    elif semantic == 'pos':
        recommended = recommended[recommended['sentiment-score'] > SEMANTIC_THRESHOLD]
    
    # filter semantic flag
    if semantic == 'neg':
        recommended = recommended[recommended['similarity_score'] < -SEMANTIC_THRESHOLD]
    elif semantic == 'pos':
        recommended = recommended[recommended['similarity_score'] > SEMANTIC_THRESHOLD]
    

    if FORMULA == 1: 
        recommended['weight-score'] = ( ((recommended['helpful_votes'].mul(HELPFUL_VOTES_MULTIPLIER)).add(recommended['total_votes'].mul(TOTAL_VOTES_MULTIPLIER))).add(1) )
        recommended['rank-score'] = ( (recommended['weight-score']).mul(recommended['sentiment-score']))
        recommended['rec-score'] = ( (recommended['rank-score']).add((recommended['similarity_score'].mul(SIMILARITY_MULTIPLIER))))
        # print("   calculated : ", recommended.head(10))

        grouped_df = recommended.groupby(['product_id', 'product_title'], as_index=False).mean()

        print("   GROUPED : \n", grouped_df.head(10))
    
    elif FORMULA == 2: 
        recommended['weight-score'] = ( ((recommended['helpful_votes'].add(recommended['total_votes'])).mul(ALL_VOTES_MULTIPLIER)).add(1) )
        recommended['rank-score'] = ( (recommended['weight-score']).mul(recommended['sentiment-score']))
        recommended['rec-score'] = ( (recommended['rank-score']).add((recommended['similarity_score'].mul(SIMILARITY_MULTIPLIER))))
        # print("   calculated : ", recommended.head(10))

        grouped_df = recommended.groupby(['product_id', 'product_title'], as_index=False).mean()

        print("   GROUPED : \n", grouped_df.head(10))

    elif FORMULA == 4: 
        recommended['weight-score'] = ( ((recommended['helpful_votes'].add(recommended['total_votes'])).mul(ALL_VOTES_MULTIPLIER)).add(1) )
        recommended['rank-score'] = ( (recommended['weight-score']).mul(recommended['sentiment-score']))
        recommended['rec-score'] = ( ((recommended['similarity_score'].mul(10)).add(recommended['rank-score'])).mul(SIMILARITY_MULTIPLIER))
        # print("   calculated : ", recommended.head(10))

        grouped_df = recommended.groupby(['product_id', 'product_title'], as_index=False).mean()

        print("   GROUPED : \n", grouped_df.head(10))

    # print(big_df.head())
    # Sort DataFrame by calculated recommended score in descending order and show top 10 rows
    top_10 = grouped_df.sort_values(by='rec-score', ascending=False)
    top_10 = top_10.head(10)
    print(top_10)

    top_10.to_csv('top10_output.csv')
    return top_10[['product_title','rec-score']],"None"

def generate_top_ten_v4(keywords = 'math science book text', semantic = 'pos', filename_list = ['2005']):

    # Load pre-trained BERT model and tokenizer
    MODEL = 'bert-base-uncased'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)

    # Encode keyword and text tokens using BERT
    encoded_keywords = tokenizer(keywords.lower(), return_tensors='pt')['input_ids']

    df_lst = []

    # Generate BERT embeddings for keyword and text tokens batch-wise and compute cosine similarity
    with torch.no_grad():
        keyword_embeddings = model(encoded_keywords.to(device))[0][:, 0, :].cpu().numpy()
    for filename in filename_list:
        # Load DataFrame
        try:
            df = pd.read_pickle('book-pickle-4/'+ filename +'.pkl')
        except(FileNotFoundError):
            continue

        df['similarity_score'] = np.nan 
        
        for idx in tqdm(range(len(df))):

            text_embeddings = df.iloc[idx]['keywords_embedding']
            # Compute cosine similarity between keyword vector and text vectors in the batch
            max_similarity_score = max(cosine_similarity(keyword_embeddings, text_embeddings)[0])
            # Add similarity scores to DataFrame
            df.at[idx, 'similarity_score'] = max_similarity_score

        df_lst.append(df)

    big_df = pd.concat(df_lst)

    # Make new copy dataframe for manipulation
    recommended = big_df[['product_id','product_title','helpful_votes', 'total_votes', 'sentiment-score']].copy()
    # recommended = big_df[['product_id','product_title','star_rating','helpful_votes', 'total_votes', 'sentiment-score']].copy()
    print(big_df['similarity_score'].head(10))

    recommended['similarity_score'] =  big_df['similarity_score'].astype('float64')
    
    # print("  INFO \n", recommended.info())
    
    # filter semantic flag
    if semantic == 'neg':
        recommended = recommended[recommended['sentiment-score'] < -SEMANTIC_THRESHOLD]
    elif semantic == 'pos':
        recommended = recommended[recommended['sentiment-score'] > SEMANTIC_THRESHOLD]
    
    # filter semantic flag
    if semantic == 'neg':
        recommended = recommended[recommended['similarity_score'] < -SEMANTIC_THRESHOLD]
    elif semantic == 'pos':
        recommended = recommended[recommended['similarity_score'] > SEMANTIC_THRESHOLD]
    

    if FORMULA == 1: 
        recommended['weight-score'] = ( ((recommended['helpful_votes'].mul(HELPFUL_VOTES_MULTIPLIER)).add(recommended['total_votes'].mul(TOTAL_VOTES_MULTIPLIER))).add(1) )
        recommended['rank-score'] = ( (recommended['weight-score']).mul(recommended['sentiment-score']))
        recommended['rec-score'] = ( (recommended['rank-score']).add((recommended['similarity_score'].mul(SIMILARITY_MULTIPLIER))))
        # print("   calculated : ", recommended.head(10))

        grouped_df = recommended.groupby(['product_id', 'product_title'], as_index=False).mean()

        print("   GROUPED : \n", grouped_df.head(10))
    
    elif FORMULA == 2: 
        recommended['weight-score'] = ( ((recommended['helpful_votes'].add(recommended['total_votes'])).mul(ALL_VOTES_MULTIPLIER)).add(1) )
        recommended['rank-score'] = ( (recommended['weight-score']).mul(recommended['sentiment-score']))
        recommended['rec-score'] = ( (recommended['rank-score']).add((recommended['similarity_score'].mul(SIMILARITY_MULTIPLIER))))
        # print("   calculated : ", recommended.head(10))

        grouped_df = recommended.groupby(['product_id', 'product_title'], as_index=False).mean()

        print("   GROUPED : \n", grouped_df.head(10))

    elif FORMULA == 4: 
        recommended['weight-score'] = ( ((recommended['helpful_votes'].add(recommended['total_votes'])).mul(ALL_VOTES_MULTIPLIER)).add(1) )
        recommended['rank-score'] = ( (recommended['weight-score']).mul(recommended['sentiment-score']))
        recommended['rec-score'] = ( ((recommended['similarity_score'].mul(10)).add(recommended['rank-score'])).mul(SIMILARITY_MULTIPLIER))
        # print("   calculated : ", recommended.head(10))

        grouped_df = recommended.groupby(['product_id', 'product_title'], as_index=False).mean()

        print("   GROUPED : \n", grouped_df.head(10))

    # print(big_df.head())
    # Sort DataFrame by calculated recommended score in descending order and show top 10 rows
    top_10 = grouped_df.sort_values(by='rec-score', ascending=False)
    top_10 = top_10.head(10)
    print(top_10)

    top_10.to_csv('top10_output.csv')
    return top_10[['product_title','rec-score']],"None"

def generate_top_ten_with_prices(keywords = 'math science book text', semantic = 'pos', filename_list = ['2005']):
    # Load pre-trained BERT model and tokenizer
    MODEL = 'bert-base-uncased'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)

    # Encode keyword and text tokens using BERT
    encoded_keywords = tokenizer(keywords.lower(), return_tensors='pt')['input_ids']

    df_lst = []

    # Generate BERT embeddings for keyword and text tokens batch-wise and compute cosine similarity
    with torch.no_grad():
        keyword_embeddings = model(encoded_keywords.to(device))[0][:, 0, :].cpu().numpy()
    for filename in filename_list:
        # Load DataFrame
        try:
            df = pd.read_pickle('book-pickle-4/'+ filename +'.pkl')
        except(FileNotFoundError):
            continue

        df['similarity_score'] = np.nan 
        
        for idx in tqdm(range(len(df))):

            text_embeddings = df.iloc[idx]['keywords_embedding']
            # Compute cosine similarity between keyword vector and text vectors in the batch
            try:
                max_similarity_score = max(cosine_similarity(keyword_embeddings, text_embeddings)[0])
            except:
                continue
            # Add similarity scores to DataFrame
            df.at[idx, 'similarity_score'] = max_similarity_score

        df_lst.append(df)

    big_df = pd.concat(df_lst)

    # Make new copy dataframe for manipulation
    recommended = big_df[['product_id','product_title','helpful_votes', 'total_votes', 'sentiment-score']].copy()
    # recommended = big_df[['product_id','product_title','star_rating','helpful_votes', 'total_votes', 'sentiment-score']].copy()
    print(big_df['similarity_score'].head(10))

    recommended['similarity_score'] =  big_df['similarity_score'].astype('float64')
    
    # print("  INFO \n", recommended.info())
    
    # filter semantic flag
    if semantic == 'neg':
        recommended = recommended[recommended['sentiment-score'] < -SEMANTIC_THRESHOLD]
    elif semantic == 'pos':
        recommended = recommended[recommended['sentiment-score'] > SEMANTIC_THRESHOLD]
    
    # filter semantic flag
    if semantic == 'neg':
        recommended = recommended[recommended['similarity_score'] < -SEMANTIC_THRESHOLD]
    elif semantic == 'pos':
        recommended = recommended[recommended['similarity_score'] > SEMANTIC_THRESHOLD]
    

    if FORMULA == 1: 
        recommended['weight-score'] = ( ((recommended['helpful_votes'].mul(HELPFUL_VOTES_MULTIPLIER)).add(recommended['total_votes'].mul(TOTAL_VOTES_MULTIPLIER))).add(1) )
        recommended['rank-score'] = ( (recommended['weight-score']).mul(recommended['sentiment-score']))
        recommended['rec-score'] = ( (recommended['rank-score']).add((recommended['similarity_score'].mul(SIMILARITY_MULTIPLIER))))
        # print("   calculated : ", recommended.head(10))

        grouped_df = recommended.groupby(['product_id', 'product_title'], as_index=False).mean()

        print("   GROUPED : \n", grouped_df.head(10))
    
    elif FORMULA == 2: 
        recommended['weight-score'] = ( ((recommended['helpful_votes'].add(recommended['total_votes'])).mul(ALL_VOTES_MULTIPLIER)).add(1) )
        recommended['rank-score'] = ( (recommended['weight-score']).mul(recommended['sentiment-score']))
        recommended['rec-score'] = ( (recommended['rank-score']).add((recommended['similarity_score'].mul(SIMILARITY_MULTIPLIER))))
        # print("   calculated : ", recommended.head(10))

        grouped_df = recommended.groupby(['product_id', 'product_title'], as_index=False).mean()

        print("   GROUPED : \n", grouped_df.head(10))

    elif FORMULA == 4: 
        recommended['weight-score'] = ( ((recommended['helpful_votes'].add(recommended['total_votes'])).mul(ALL_VOTES_MULTIPLIER)).add(1) )
        recommended['rank-score'] = ( (recommended['weight-score']).mul(recommended['sentiment-score']))
        recommended['rec-score'] = ( ((recommended['similarity_score'].mul(10)).add(recommended['rank-score'])).mul(SIMILARITY_MULTIPLIER))
        # print("   calculated : ", recommended.head(10))

        grouped_df = recommended.groupby(['product_id', 'product_title'], as_index=False).mean()

        print("   GROUPED : \n", grouped_df.head(10))

    # print(big_df.head())
    # Sort DataFrame by calculated recommended score in descending order and show top 10 rows
    top_10 = grouped_df.sort_values(by='rec-score', ascending=False)
    top_10 = top_10.head(10)
    
    top_10_ids = top_10['product_id'].to_list()
    try:
        top_10_prices = get_price_from_list_IDs(top_10_ids)
        top_10 = top_10.join(top_10_prices, on='product_id')
    except:
        print(top_10)
        return top_10[['product_title','rec-score']],"Error in fetching product prices, showing result without price information"
    print(top_10)
    top_10.to_csv('top10_output.csv')
    return top_10[['product_title','rec-score','price']],"None"

if __name__ == "__main__":
    generate_top_ten_with_prices()