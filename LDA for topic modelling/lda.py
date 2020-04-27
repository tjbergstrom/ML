# topic modelling

# usage:
# python3 lda.py

# what this does:
# takes input text from a .txt file
# assuming that text has already been filtered down to the actual articles
# and assuming that they are already filtered for keyword "alaska"
# preprocesses that text (like stopwords, punctuation, stemming, etc)
# makes an LDA model with the processed text
# the model generates 5 topics represented by 10 words each
# these topics are printed to terminal output
# the first 4 topics are also generated into a word cloud, and the cloud is saved
# the 10 most common words are also plotted just for fun, this plot is also saved

# unfinished - trying to save the cloud and plot to a pdf
# unfinished - LDA generated topics are just represented by 10 words,
#  so a human has to interpret what that topic is,
#  so next thing to do is work on guessing a topic from a list of most likely topics,
#  then save the topic in order to compare trending topics from week to week
# unfinished - testing with better input text to build the best model,
#  ie get the most accurate topic predictions


import nltk
import urllib.request
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import PyPDF2
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import string
from nltk.stem.porter import PorterStemmer
import os
import pandas as pd
import re
# from sklearn.feature_extraction.text import CountVectorize
from sklearn import datasets
from nltk.corpus import stopwords
# from wordcloud import wordcloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %matplotlib inline
import gensim
from gensim import corpora
from pprint import pprint
from gensim.utils import simple_preprocess
from smart_open import smart_open
import os
from gensim.models import LdaModel, LdaMulticore
import gensim.downloader as api
from gensim.utils import simple_preprocess, lemmatize
from nltk.corpus import stopwords
import re
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# import pyLDAvis
# import pyLDAvis.gensim
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import glob
import pathlib
# nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer("english")
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import common_corpus
from nltk import word_tokenize
from nltk import bigrams, trigrams
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from PIL import Image


########################################
# helper functions and print functions #
########################################


'''
# option to change input file with command argument
# python3 lda_work.py --file other_file.txt
import argparse # pip3 install argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--file", default="articles.txt", help="path to input file")
args = vars(ap.parse_args())
input_file = args["file"]
'''
input_file = "articles.txt"

def plot_10_most_common_words(count_data, count_vectorizer):
    print("\n\ninfo: plotting ten most common words...")
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))
    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    #plt.show()
    plt.savefig('tenwords.png', bbox_inches='tight')
    print("info: saved ten most common words plot to disk\n\n")


def print_topics(model, count_vectorizer, n_top_words):
    print("\n\ninfo: printing topics: \n\n")
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        #print("\nTopic #%d:" % topic_idx)
        print("\nTopic:",topic_idx+1)
        print(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))


def make_topics(model, count_vectorizer, n_top_words):
    topic_words = []
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        topic_words.append(" ".join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    return topic_words


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


education = ["education", "university", "school"]
transportation = ["ferry", "road", "bus"]
outdoors = ["hunt"]
money = ["money", "budget", "dollars", "fiscal"]
climate = ["temperatures", "arctic", "climate"]
oil = ["oil"]


def guess_topic(words):
    topic = ''
    for word in words:
        # print(word)
        if word in education:
            topic += "education "
        if word in transportation:
            topic += "education "
        if word in outdoors:
            topic += "outdoors "
        if word in money:
            topic += "money "
        if word in oil:
            topic += "oil "
    if topic == "":
        topic = "unknown"
    return topic


def make_cloud(lda):
    print("\n\ninfo: making wordcloud...")
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
    cloud = WordCloud(stopwords=sw,
                      background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)
    topics = make_topics(lda, count_vectorizer, number_words)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        # topic_words = dict(topics[i])
        topic_words = topics[i]
        cloud.generate(topic_words)
        # cloud.generate(topics)
        plt.gca().imshow(cloud)
        topic = ""
        words = []
        word = ""
        for t in topics[i]:
            if t == " ":
                words.append(word)
                word = ""
                continue
            word += t
        # print(words)
        topic = guess_topic(words)
        # plt.gca().set_title("topic " + str(i), fontdict=dict(size=16))
        plt.gca().set_title("topic best guess = " + topic, fontdict=dict(size=16))
        plt.gca().axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    #plt.show()
    plt.savefig('cloud.png', bbox_inches='tight')
    print("info: wordcloud saved to disk\n\n")


#####################################
# preprocessing text from .txt file #
#####################################


print("\n\ninfo: preprocessing...\n\n")


file_open = open(input_file, encoding="utf8")
file_read = file_open.read()
# read all text into a string
text = ''
for word in file_read.split():
    text = text + " " + word
# split text into tokens
soup = BeautifulSoup(text, "html5lib")
txt = soup.get_text(strip=True)
clean = txt.lower()
tokens = [t for t in clean.split()]
# clean up stop words
clean_tokens = tokens[:]
sw = stopwords.words('english')
sw.extend(['say', 'subject', 're', 'edu', 'use', 'said', 'says', 'they', 'it', 'we', 'one', 'could', 'would'])
sw.extend(['jr', 'trump', 'year', 'years', 'last', 'eat', 'eaten', 'donald', 'feb', 'couple', " "])
for token in tokens:
    if token in sw:
        clean_tokens.remove(token)
# clean up punctuation
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in clean_tokens]
# extra remove stop words, does better after punctuation
for token in stripped:
    if token in sw:
        stripped.remove(token)
sep = " "
n = 3
grams = [sep.join(stripped[i:i + n]) for i in range(len(stripped) - n + 1)]
# stemming
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in stripped]
# print most used words
#freq = nltk.FreqDist(stemmed)
#freq = nltk.FreqDist(stripped)
#freq.plot(20, cumulative=False)


###########################
#####  LDA modelling  #####
###########################


print("\n\ninfo: making LDA model...\n\n")


# pass preprocessed text to LDA library functions
count_vectorizer = CountVectorizer(sw)
# tuning this with different parameters
# count_vectorizer = CountVectorizer(max_df = 0.95, min_df = 2, stop_words)
# count_data = count_vectorizer.fit_transform(stemmed)
# count_data = count_vectorizer.fit_transform(clean_tokens)
count_data = count_vectorizer.fit_transform(stripped)
# plot most common words
plot_10_most_common_words(count_data, count_vectorizer)
number_topics = 5
number_words = 10
# feed into an LDA model
# original model
# lda = LDA(n_components=number_topics, n_jobs=-1)
# maybe this model looks better
lda = LatentDirichletAllocation(n_components=number_topics, random_state=42)
# this model does pretty well
# all these parameters can be tuned to get a better model
#lda = LatentDirichletAllocation(n_components=number_topics, max_iter=20,
#                                learning_method='online',
#                                learning_decay=0.9,  # default 0.7
#                                learning_offset=50.,
#                                evaluate_every=5,  # default 0
#                                random_state=0)
lda.fit(count_data)
print_topics(lda, count_vectorizer, number_words)
make_cloud(lda)


############################################
# end of LDA work
# trying to save the trending themes to file
############################################


from fpdf import FPDF
from datetime import datetime
from datetime import date
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
today = str(datetime.date(datetime.now()))
today = str(date.today())
title = "Trending themes for week of "
title += today
#topix = print_topics(lda, count_vectorizer, number_words)
pdf.cell(200, 10, txt=title, ln=1, align="C")
#pdf.cell(txt=topix)
print("\n\ninfo: saving trending themes to file")
pdf.image("cloud.png")
pdf.image("tenwords.png")
pdf.output("testings.pdf")
print("info: saved trending themes to file\n\n")


# old ignore
# other LDA models from different libraries
# tfidf, tf, nmf, and nmf frobenious models
# doesn't seem like they generate topics as well
# maybe need to be tuned more
'''
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=1000,
                                   stop_words=sw)
tfidf = tfidf_vectorizer.fit_transform(stripped)
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=1000,
                                stop_words=sw)
tf = tf_vectorizer.fit_transform(stripped)
nmf = NMF(n_components=number_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
# print("NMF with Frobenius norm")
print_topics(nmf, count_vectorizer, number_words)
make_cloud(nmf)

# In[20]:


nmf = NMF(n_components=number_topics, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)
print_topics(nmf, count_vectorizer, number_words)
make_cloud(nmf)

# In[21]:


lda.fit(tf)
print_topics(lda, count_vectorizer, number_words)
make_cloud(nmf)

# In[ ]:
'''

print("\n\ninfo: task failed successfully\n\n")



