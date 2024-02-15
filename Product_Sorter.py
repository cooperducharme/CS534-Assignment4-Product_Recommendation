from pickle import GET
from functions import *
from classes import *
import os
import openai
import json
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pickle
import torch.optim as optim

net = Net(30,15,1)

use_processed_products = False

async def main():
    #print(await get_price_range('pressure_cooker'))

    if use_processed_products == False:
        review_list = import_pickle('2005.pkl')
        for review in review_list:
            if review.similarity_score!=0:
                print(review)
        product_list = await create_product_list(review_list)
        for product in product_list:
            if len(product.review_list) > 1:
                product.review_list = sort_reviews_by_helpful(product.review_list)
        product_list = scale_reviews(product_list)
        for product in product_list:
            product.average_rating = calculate_average_stars(product.review_list)
            product.average_similarity = calculate_average_similarity(product.review_list)
        
        pickle_file = 'processed_reviews.pk'
    
        with open(pickle_file, 'wb') as fi:
            pickle.dump(product_list, fi)

    else:
        pickle_file = 'processed_reviews.pk'
        with open(pickle_file, 'rb') as fi:
            product_list = pickle.load(fi)
     
    product_parameters = []
    product_values = []
    for product in product_list:
        tensor1,tensor2 = convert_To_Tensors(product)
        product_parameters.append(tensor1)
        product_values.append(tensor2)

    



    train_model(product_parameters,product_values,net)

    #product_list = sort_products(product_list,net)

    best_product = find_top_product(product_list,net)
    print(best_product)


 

    

if __name__ == "__main__":
    asyncio.run(main())


review_list = import_csv('example-2005.csv')

