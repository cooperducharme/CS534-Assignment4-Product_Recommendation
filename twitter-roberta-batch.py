from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
import torch
import pandas as pd
from scipy.special import softmax
import csv
import urllib.request
from tqdm import tqdm

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
 
 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
# 
# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

list_of_names = ['2005.csv','2004.csv','2003.csv','2002.csv','2001.csv','2000.csv']
for each_filename in list_of_names:
    df = pd.read_csv('book-csv/'+each_filename)
    # df = df.dropna()
    # # Combine review headline and review body into a single column
    # df['clean_text'] = df['review_headline'] + ' ' + df['review_body']
    # df = df[['product_id', 'product_title', 'star_rating', 'helpful_votes', 
    #           'total_votes', 'vine', 'verified_purchase', 'clean_text']]
    all_scores = []
    # define batch size and number of batches
    batch_size = 256
    num_batches = (len(df) + batch_size - 1) // batch_size
    
    # loop over all batches
    for i in tqdm(range(num_batches)):
        # get start and end indices for the current batch
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        
        # select the texts for the current batch
        batch_texts = df['clean_text'][start_idx:end_idx].tolist()

        # encode the texts using the tokenizer
        encoded_inputs = tokenizer(batch_texts, return_tensors='pt',padding=True, truncation=True, max_length=512).to(device)
        
        # get the output from the model
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        
        # calculate the scores and convert to numpy arrays
        batch_scores = outputs[0].detach().cpu().numpy()
        batch_scores_softmax = np.apply_along_axis(lambda row: np.exp(row) / np.sum(np.exp(row)), axis=1, arr=batch_scores)
        # print(batch_scores_softmax)
        # calculate the final score for each text in the batch
        batch_final_scores = batch_scores_softmax[:, 2] - batch_scores_softmax[:, 0]
        
        # append the final scores to the list
        all_scores.extend(batch_final_scores.tolist())

    # convert the list of scores to a numpy array and add as columns to the dataframe
    score_columns = ['sentiment-score']
    score_array = np.array(all_scores)
    df[score_columns] = pd.DataFrame(score_array, columns=score_columns)

    df.to_csv('book-sentiment/'+each_filename, index = None, header=True) 

# text = "Good night 😊"
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
# scores = output[0][0].detach().numpy()
# scores = softmax(scores)
# print(scores)
# ranking = np.argsort(scores)
# ranking = ranking[::-1]
# for i in range(scores.shape[0]):
#     l = labels[ranking[i]]
#     s = scores[ranking[i]]
#     print(f"{i+1}) {l} {np.round(float(s), 4)}")