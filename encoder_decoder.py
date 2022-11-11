import re
from nltk.tokenize import sent_tokenize
import contractions
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

DATASET_LENGTH = 1000
dataset = load_dataset("multi_news")

#Text Preprocessing
def parse_article(article):
    expanded_words = []     #expand contractions
    for words in article.split():
        expanded_words.append(contractions.fix(words))
    article = " ".join(expanded_words)
    sent_list = sent_tokenize(article)
    clean_sent = [re.sub("\W+", " ",sent).lower() for sent in sent_list]   #removes punctuation and makes all words lower case
    return clean_sent

ds = dataset['train'][:DATASET_LENGTH]
x_train,x_test,y_train,y_test = train_test_split(ds['document'],ds['summary'], test_size=0.2,shuffle=True)


def embedding_matrix():
    #Get pretrained embeddings
    embeddings_wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
    embeddings_wv.init_sims(replace=True)

    #Get all unique words
    unique_words = set()
    for i in x_train :
        clean = parse_article(i)
        for sent in clean:
            for word in sent.split():
                unique_words.add(word)

    #Create our own embedding matrix with words in our articles   
    wordIndex = {list(unique_words)[i]:i for i in range(len(list(unique_words)))}
    vocab_size = len(unique_words)
    embedding_matrix = np.zeros((vocab_size,300))

    #if word is in pretrained embedding, take that embedding and add it to our matrix
    for word,index in wordIndex.items():
        try:
            embedding_vector = embeddings_wv[word]
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
        except:
            pass
    return embedding_matrix





# print(parse_reviews(dataset['train'][1]['document']))
# print(dataset['train'][1]['document'])