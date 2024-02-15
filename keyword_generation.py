import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import torch

def get_keybert_results_with_vectorizer(list_of_text, number_of_results=5,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                        ):

    sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    kw_model = KeyBERT(model=sentence_model)
    list_of_keywords = []
    for text in list_of_text:
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3),
                    stop_words='english',use_maxsum=True)
        keywords = [i for i in keywords if i[1] > 0.20]

        keybert_diversity_phrases = []
        for i, j in keywords:
            keybert_diversity_phrases.append(i)

        if len(keybert_diversity_phrases) > number_of_results:
            keybert_diversity_phrases = keybert_diversity_phrases[:number_of_results]
        list_of_keywords.append(keybert_diversity_phrases)
    return list_of_keywords

def get_keybert_results_with_vectorizer_df(list_of_text, number_of_results=5, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    kw_model = KeyBERT(model=sentence_model)
    
    df = pd.DataFrame({'text': list_of_text})
    df['keywords'] = df['text'].apply(lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(1, 3), stop_words='english', use_maxsum=True))
    df['keywords'] = df['keywords'].apply(lambda x: [i for i in x if i[1] > 0.20])
    df['keywords'] = df['keywords'].apply(lambda x: [i[0] for i in x])
    df['keywords'] = df['keywords'].apply(lambda x: x[:number_of_results] if len(x) > number_of_results else x)
    
    return df['keywords'].tolist()

if __name__ == "__main__":
    print(get_keybert_results_with_vectorizer(["""
Excellent introduction to solar sails &quot;Space Sailing&quot; by Wirght is an excellent introduction to the concept of solar sails. The first two chapters introduce the concept of space sailing and provides examples of possible uses. Later chapters go into more details on ship design, sail materials and construction, ship deployment issues, and ship control mechanisms. I would recommend this book to anyone interested in space sails.  
""",
"""In this example, we initialize a KeyBERT model with the distilbert-base-nli-mean-tokens pre-trained model and the PyTorch backend on the device (either cuda or cpu). If the device is cuda, the model will automatically use CUDA acceleration."""
]))