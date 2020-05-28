#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sqlalchemy as db
import tweepy
import re
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import learning_curve
from datetime import datetime, timedelta


# ## Fetching Tweets

# In[27]:


def get_tweets(keyword):
    
    credentials = pd.read_csv('/Users/joaootaviomeirellesratton/desktop/credentials.csv')
    
    authenticate = tweepy.OAuthHandler(credentials['key'][0], credentials['key'][1])
    authenticate.set_access_token(credentials['key'][2], credentials['key'][3])
    api = tweepy.API(authenticate, wait_on_rate_limit = True)
    
    end_date = datetime.strftime(datetime.now(), '%Y-%m-%d')
    start_date = datetime.strftime(datetime.now() - timedelta(7), '%Y-%m-%d')
    
    try:
        tweets = api.search(keyword, lang='en', count=100, since=start_date, until=end_date)
        
        print('Fetched ' + str(len(tweets)) + ' tweets for the string: ' +  keyword)
        
        txt_list = [result.text for result in tweets]

        return list(map(preprocess_tweet_text, txt_list))
    
    except:
        return print('Something went wrong...')


# In[28]:


user_imput = input('Enter a search keyword: ')
target_tweets = get_tweets(user_imput)


# ## Train Dataset

# In[2]:


db_server = 'postgresql'
user = 'postgres'
password = '180592'
ip = 'localhost'
db_name = 'twitter_trainset'

engine = db.create_engine(f'{db_server}://{user}:{password}@{ip}/{db_name}')

conn = engine.connect()


# In[3]:


query ="""
SELECT *
FROM tweets;
"""
train_set_full = pd.read_sql_query(query, con=conn)


# In[4]:


train_set_final = train_set_full.sample(n=15000, random_state=42)


# ## Preprocessing Text

# In[5]:


def preprocess_tweet_text(tweet):

    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#','', tweet)
    tweet = re.sub(r'^RT', '', tweet)
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = re.sub(r'[0-9]+', '', tweet)
    
    return tweet


# In[6]:


train_set_final['tweet'] = train_set_final['tweet'].apply(preprocess_tweet_text)


# In[7]:


train_array = train_set_final['tweet'].tolist()


# In[8]:


regexp = re.compile('(?u)\\b\\w\\w+\\b')
en_nlp = spacy.load('en', disable=['parser', 'ner'])
old_tokenizer = en_nlp.tokenizer
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(regexp.findall(string))


# In[9]:


def custom_tokenizer(document):
    
    doc_spacy = en_nlp(document)  
    return [token.lemma_ for token in doc_spacy]


# In[10]:


lemma_vect = TfidfVectorizer(tokenizer=custom_tokenizer, min_df=5, ngram_range=(1, 2))


# In[11]:


X_train_lemma = lemma_vect.fit_transform(train_array)


# In[18]:


y = train_set_final['sentment']


# ## Model

# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X_train_lemma, y, random_state=42)


# In[20]:


clf = naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)


# In[21]:


roc_auc_score(y_test, clf.predict_proba(X_test)[:,1], multi_class='ovr')


# In[22]:


curve = learning_curve(clf, X_train_lemma, y)
train_sizes = curve[0]
train_scores = curve[1]
test_scores = curve[2]


# In[23]:


plt.plot(train_sizes, train_scores.mean(axis=1), '-o', label='Training AUC')
plt.plot(train_sizes, test_scores.mean(axis=1), '-o', label='Test AUC')
plt.ylim([0., 1])
plt.xlabel('Training Sizes')
plt.ylabel('Scores')
plt.legend(loc=4)
plt.grid()


# ## Test

# In[34]:


def classify_tweets(tweets):
    
    tweets_vector = lemma_vect.transform(target_tweets)
    result_labels = clf.predict(tweets_vector)
    
    return print(
        'Overall negative sentiment: {:.1%}'.format(np.count_nonzero(result_labels == 0) / len(result_labels)),
        'Overall positive sentiment: {:.1%}'.format(np.count_nonzero(result_labels == 4) / len(result_labels)), sep='\n')


# In[35]:


classify_tweets(target_tweets)


# In[45]:


def classify_tweets(tweets):
    
    tweets_vector = lemma_vect2.transform(text)
    result_labels = clf.predict(text)
    
    return print(
        'Overall negative sentiment: {:.1%}'.format(np.count_nonzero(result_labels == 0) / len(result_labels)),
        'Overall positive sentiment: {:.1%}'.format(np.count_nonzero(result_labels == 4) / len(result_labels)), sep='\n')


# In[306]:


import pickle


# In[307]:


pickle.dump(clf, open('modelo_final.sav', 'wb'))


# In[308]:


loaded_model = pickle.load(open('modelo_final.sav', 'rb'))


# In[310]:


pickle.dump(lemma_vect, open('modelo_lemm.sav', 'wb'))


# In[311]:


loaded_class = pickle.load(open('modelo_lemm.sav', 'rb'))


# In[312]:


loaded_class.transform(tweets_array)

