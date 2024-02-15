from requests_html import AsyncHTMLSession
from EdgeGPT import Chatbot, ConversationStyle
from classes import *
import csv
import datetime
import sqlite3
import os
import openai
import json
import asyncio
import pickle
import pandas as pd
import nest_asyncio
import pyterrier as pt
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

verbose_import_csv = False
verbose_price_check = False

# import_csv 
def import_csv(file_name):
    review_list = []
    with open(file_name, 'r') as file:
        reader = csv.reader(file,delimiter=',')
        line_num = 0
        for row in reader:
            if line_num == 0:
                if verbose_import_csv == True:
                    print(f'Column labels are {",".join(row)}')
                line_num+=1
            else:
                if verbose_import_csv == True:
                    print(f'product ID:{row[0]},product title:{row[1]},review date:{row[2]},star rating:{row[3]},helpful upvotes:{row[4]},total votes:{row[5]},vine:{[6]},verified purchase{[7]},clean text{[8]},sentiment score{row[9]},similarity score{row[10]}')
                review_list.append(Review(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10]))
    return review_list

# split_string_in_half
def split_string_in_half(string):
    half = ''
    for i in range(0,int(len(string)/2)):
        half = half+string[i]
    return half

# price_check
async def price_check(ID):
    nest_asyncio.apply()
    s = AsyncHTMLSession()
    asin = ID
    r = await s.get(f'https://www.amazon.com/dp/{asin}')
    await r.html.arender(sleep=1)
    price1 = r.html.find('#tp_price_block_total_price_ww')
    price2 = r.html.find('#tp-tool-tip-subtotal-price-value')
    if price2:
        price = price2[0].text.replace('$','').replace('.','').strip()
        price = int(split_string_in_half(price))/100
        if verbose_price_check:
            print(f'product ID:{asin} costs {price}')
        return price
    elif price1:
        price = price1[0].text.replace('$','').replace(',','').strip()
        price = int(split_string_in_half(price))/100
        if verbose_price_check:
            print(f'product ID:{asin} costs {price}')
        return price
    else:
        if verbose_price_check:
            print(f'product ID:{asin} is out of stock. No price available')
        return -1

async def get_price_range(search_term):
    with open('cookies.json', 'r') as f:
        cookies = json.load(f)
    bot = Chatbot(cookies=cookies)
    response = await bot.ask(prompt=f'how much should a person pay for a {search_term}',  wss_link="wss://sydney.bing.com/sydney/ChatHub")
    print(response['item']['messages'][1]['text'])
    response = str(response)
    dollar_sign_list = []
    price_list = []
    for i in range(0,len(response)):
        if response[i] == '$':
            dollar_sign_list.append(i)
    for x in dollar_sign_list:
        index = x+1
        if (response[index].isnumeric()):
            price = ''
            while response[index].isnumeric():
                price = price+response[index]
                index+=1
            price_list.append(int(price))
    print(price_list)
    mean = sum(price_list) / len(price_list)
    variance = sum([((x - mean) ** 2) for x in price_list]) / len(price_list)
    res = variance ** 0.5
    await bot.close()
    return mean,res

def import_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    review_list = []
    for index, row in data.iterrows():
        print(row)
        review_list.append(Review(row['product_id'],row['product_title'],row['review_date'],row['star_rating'],row['helpful_votes'],row['total_votes'],row['vine'],row['verified_purchase'],row['clean_text'],row['sentiment-score'],row['similarity_score']))
    return review_list

def sort_list(review_list):
    sortd = False
    while not sortd:
        i = 1
        while i < (len(review_list)-1):
            if review_list[i].product_id > review_list[i+1].product_id:
                review_list[i], review_list[i+1] = review_list[i+1], review_list[i]
            i+=1
            if i == len(review_list):
                sortd = True
    return review_list
            
async def create_product_list(review_list):
    product_list = []
    reviews = 0
    for review in review_list:
        reviews+=1
        product_exists = False
        for product in product_list:
            if product.product_id == review.product_id:
                product.review_list.append(review)
                product_exists = True
        if not product_exists:
            # price = await price_check(review.product_id)
            review_list_p = []
            review_list_p.append(review)
            price = 0
            product_list.append(Product(review.product_id,review.product_title,review.star_rating,price,review_list_p,0))
    return product_list

def sort_reviews_by_helpful(review_list):
    sortd = False
    while not sortd:
        i = 0
        while i < (len(review_list)-1):
            if review_list[i].helpful_votes < review_list[i+1].helpful_votes:
                review_list[i], review_list[i+1] = review_list[i+1], review_list[i]
                i = 0
                sortd = False
            else:
                i+=1
                sortd = True
    return review_list

def calculate_average_stars(review_list):
    total = 0
    for review in review_list:
        total+=review.star_rating
    return total/10

def calculate_average_similarity(review_list):
    total = 0
    for review in review_list:
        total+=review.similarity_score
    return total/10


def scale_reviews(product_list):
    for product in product_list:
        if len(product.review_list)<10:
            number_reviews_needed = 10-len(product.review_list)
            for i in range(0,number_reviews_needed):
                product.review_list.append(product.review_list[i])
        elif len(product.review_list)>10:
            while len(product.review_list)>10:
                product.review_list.pop()
    return product_list

def convert_To_Tensors(product):
    tensor1 = [product.review_list[0].sentiment_score,product.review_list[0].similarity_score,product.review_list[0].helpful_votes,
                  product.review_list[1].sentiment_score,product.review_list[1].similarity_score,product.review_list[1].helpful_votes,
                  product.review_list[2].sentiment_score,product.review_list[2].similarity_score,product.review_list[2].helpful_votes,
                  product.review_list[3].sentiment_score,product.review_list[3].similarity_score,product.review_list[3].helpful_votes,
                  product.review_list[4].sentiment_score,product.review_list[4].similarity_score,product.review_list[4].helpful_votes,
                  product.review_list[5].sentiment_score,product.review_list[5].similarity_score,product.review_list[5].helpful_votes,
                  product.review_list[6].sentiment_score,product.review_list[6].similarity_score,product.review_list[6].helpful_votes,
                  product.review_list[7].sentiment_score,product.review_list[7].similarity_score,product.review_list[7].helpful_votes,
                  product.review_list[8].sentiment_score,product.review_list[8].similarity_score,product.review_list[8].helpful_votes,
                  product.review_list[9].sentiment_score,product.review_list[9].similarity_score,product.review_list[9].helpful_votes]
    tensor2 = [product.average_rating*product.average_similarity]
    tensor1 = torch.tensor(tensor1)  
    tensor2 = torch.tensor(tensor2)  
    return tensor1,tensor2

def convert_To_Tensor(product):
    tensor1 = [product.review_list[0].sentiment_score,product.review_list[0].similarity_score,product.review_list[0].helpful_votes,
                  product.review_list[1].sentiment_score,product.review_list[1].similarity_score,product.review_list[1].helpful_votes,
                  product.review_list[2].sentiment_score,product.review_list[2].similarity_score,product.review_list[2].helpful_votes,
                  product.review_list[3].sentiment_score,product.review_list[3].similarity_score,product.review_list[3].helpful_votes,
                  product.review_list[4].sentiment_score,product.review_list[4].similarity_score,product.review_list[4].helpful_votes,
                  product.review_list[5].sentiment_score,product.review_list[5].similarity_score,product.review_list[5].helpful_votes,
                  product.review_list[6].sentiment_score,product.review_list[6].similarity_score,product.review_list[6].helpful_votes,
                  product.review_list[7].sentiment_score,product.review_list[7].similarity_score,product.review_list[7].helpful_votes,
                  product.review_list[8].sentiment_score,product.review_list[8].similarity_score,product.review_list[8].helpful_votes,
                  product.review_list[9].sentiment_score,product.review_list[9].similarity_score,product.review_list[9].helpful_votes]
    tensor1 = torch.tensor(tensor1)    
    return tensor1

def train_model(product_parameters,product_values,net):
    learning_rate = 0.001
    
    criterion = nn.L1Loss()
    optimizer = optim.SGD(net.parameters(), learning_rate, momentum=0.9)
    running_loss = 0.0
    for i in range(len(product_parameters)):
        optimizer.zero_grad()
        out = net(product_parameters[i])
        loss = criterion(out,product_values[i])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('Iteration: %5d, loss: %.3f' %
                  (i + 1, running_loss / 2000))
            running_loss = 0.0

def sort_products(product_list, net):
    sortd = False
    percent_done = 0
    while not sortd:
        i = 1
        while i < (len(product_list)-1):
            if net(convert_To_Tensor(product_list[i])) > net(convert_To_Tensor(product_list[i+1])):
                product_list[i], product_list[i+1] = product_list[i+1], product_list[i]
                i = 0
                sortd = False
            i+=1
            if i/len(product_list)*100 > percent_done:
                percent_done = i/len(product_list)*100
            if percent_done % 5 == 0:
                percent_done+=1
                print('%.3f percent done sorting' % percent_done)
            if i == len(product_list):
               sortd = True
    return product_list

def find_top_product(product_list,net):
    best_product = product_list[0]
    for product in product_list:
        if net(convert_To_Tensor(product)) > net(convert_To_Tensor(best_product)):
            best_product = product
            print( net(convert_To_Tensor(product)))
    return product

# call the function
async def run_price_check(ID):
    price = await price_check(ID)
    return price

def get_price_from_list_IDs(product_ids = ['B08XGDN3TZ', 'B07VGRJDFY', 'B09F7JHHK4']):
    # run the coroutines asynchronously
    loop = asyncio.get_event_loop()
    
    # create a list of coroutines
    # coros = [run_price_check(product_id) for product_id in product_ids]

    # tasks = [loop.create_task(coro) for coro in coros]
    prices = []
    for id in product_ids:
        task = loop.create_task(run_price_check(id))
        loop.run_until_complete(asyncio.wait([task]))
        price = task.result()
        prices.append(price)
    #  = loop.run_until_complete(asyncio.wait(tasks))

    result = pd.DataFrame({'id': product_ids, 'price':prices})
    print(result)
    return result

if __name__ == "__main__":
    get_price_from_list_IDs()
