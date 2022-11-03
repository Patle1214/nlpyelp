import pandas as pd 
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

#This part of the code reads in the json file and extracts all of the text. Then stores in a list called textreviews.

textreviews = []
with open("yelp_academic_dataset_review.json","rb") as f:
    reviews = f.readlines()
    for i in range(10000):
        stringtext = json.loads(reviews[i].decode())["text"]
        textreviews.append(stringtext)
# print(textreviews)

# #-------------------------------------------------------------#
# # Extract word vectors
# word_embeddings = {}
# f = open('glove.6B.100d.txt', encoding='utf-8')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     word_embeddings[word] = coefs
# f.close()
# #-------------------------------------------------------------#


def parse_reviews(review):
    sent_list = sent_tokenize(review)
    clean_sent = [re.sub("\W+", " ",sent).lower() for sent in sent_list]   #removes punctuation and makes all words lower case
    return clean_sent


stop_words = stopwords.words('english')
def sentence_similarity(review):
    vectorizer = CountVectorizer(stop_words='english') # add back stopwords later
    vect1 = vectorizer.fit_transform(review)
    return cosine_similarity(vect1,vect1) 
    
def generate_summary(review, top_n=6):
    ss = sentence_similarity(parse_reviews(review))
    sentence_similarity_graph = nx.from_numpy_array(ss)
    scores = nx.pagerank(sentence_similarity_graph)
    scoreindex = [k for k, v in sorted(scores.items(), key=lambda item: item[1])]
    print(scoreindex)
    for i in range(top_n): 
        print(sent_tokenize(review)[scoreindex[i]])
    
 
    
    

print(textreviews[1035])
generate_summary(textreviews[1035])
