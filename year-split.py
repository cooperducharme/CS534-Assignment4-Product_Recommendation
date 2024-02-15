import pandas as pd
from tqdm import tqdm

df = pd.read_csv('archive/amazon_reviews_us_Books_v1_02.tsv', sep='\t', error_bad_lines=False)
df = df.dropna()

# Combine review headline and review body into a single column
df['clean_text'] = df['review_headline'] + ' ' + df['review_body']
df = df[['product_id', 'product_title','product_parent','review_date', 'star_rating', 'helpful_votes', 
          'total_votes', 'vine', 'verified_purchase', 'review_headline','review_body','clean_text']]
mask0 = (df['review_date'] >= '2000-1-1') & (df['review_date'] <= '2000-12-31')
mask1 = (df['review_date'] >= '2001-1-1') & (df['review_date'] <= '2001-12-31')
mask2 = (df['review_date'] >= '2002-1-1') & (df['review_date'] <= '2002-12-31')
mask3 = (df['review_date'] >= '2003-1-1') & (df['review_date'] <= '2003-12-31')
mask4 = (df['review_date'] >= '2004-1-1') & (df['review_date'] <= '2004-12-31')
mask5 = (df['review_date'] >= '2005-1-1') & (df['review_date'] <= '2005-12-31')

masks = [mask0,mask1,mask2,mask3,mask4,mask5]
year = 2000
for the_mask in tqdm(masks):
    cur_df = df.loc[the_mask]
    cur_df.to_csv(("book-csv-2/"+str(year)+".csv"), index = None)
    year += 1
