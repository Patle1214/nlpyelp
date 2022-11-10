from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from string import punctuation
from rouge_evaluation import rouge_score

dataset = load_dataset("multi_news")
# print(dataset.shape) --> {'train': (44972, 2), 'validation': (5622, 2), 'test': (5622, 2)}
# dataset['train'][0]['document']
# dataset['train'][0]['summary']

def summarize(review, n = 3):
    stops = list(set(stopwords.words("english"))) + list(punctuation)
    sent_list = sent_tokenize(review)
    vectorizer = TfidfVectorizer(stop_words = stops)
    trsfm = vectorizer.fit_transform(sent_list)
    similarities = cosine_similarity(trsfm,trsfm)
    avgs = []
    for i in similarities:
        avgs.append(i.mean())
    sorted_sim = sorted(list(enumerate(avgs)),key = lambda x: x[1], reverse = True)
    summary = ""
    for i in range(n):
        summary += sent_list[sorted_sim[i][0]].replace("\\n"," ") + " "
    return summary
    
prediction = summarize(dataset['train'][0]['document'])
reference = dataset['train'][0]['summary']
print("Cosine:", prediction)
print()
print("Test:", reference)
print(rouge_score(prediction, reference,True))
