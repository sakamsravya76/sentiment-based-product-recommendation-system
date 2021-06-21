from flask import Flask, render_template, request
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn import *
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
import re
from pandas import DataFrame
import warnings
import joblib
import nltk


nltk.download('stopwords')
warnings.filterwarnings("ignore")


def SentimentAnalysis(df):
    df['reviews_title'] = df['reviews_title'].fillna('')
    df['user_reviews'] = df[['reviews_title', 'reviews_text']].agg('. '.join, axis=1).str.lstrip('. ')

    def striphtml(data):
        p = re.compile('<.*?>')  # Find this kind of pattern
        # print(p.findall(data))#List of strings which follow the regex pattern
        return p.sub('', data)  # Substitute nothing at the place of strings which matched the patterns

    def strippunc(data):
        p = re.compile(r'[?|!|\'|"|#|.|,|)|(|\|/|~|%|*]')
        return p.sub('', data)

    stop = stopwords.words('english')  # All the stopwords in English language
    # excluding some useful words from stop words list as we doing sentiment analysis
    excluding = ['against', 'not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
                 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shouldn', "shouldn't", 'wasn',
                 "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    stop = [words for words in stop if words not in excluding]

    snow = SnowballStemmer('english')  # initialising the snowball stemmer

    def preprocessText(text, stem=False):
        filtered_sentence = []
        final_string = []
        # print(text)
        text = striphtml(text)  # --- remove HTML Tags
        text = strippunc(text)  # --- remove Punctuation
        for w in text.split():  # returns “False”.
            if w.isalpha() and (len(w) > 2):  # --- Check is value is not numeric and has length > 2
                if w.lower() not in stop:  # --- Check if it is a stopword
                    if stem:
                        s = (snow.stem(w.lower())).encode('utf8')  # --- Stemming the word using snowball stemmer
                    else:
                        s = (w.lower()).encode('utf8')  # --- Stemming the word using snowball stemmer
                    filtered_sentence.append(s)
                else:
                    continue
            else:
                continue
        cleanedtext = b" ".join(filtered_sentence)  # string of cleaned words
        final_string.append(cleanedtext)
        return final_string

    df['cleanedReview'] = df['user_reviews'].map(preprocessText)
    df['user_reviews'] = pd.DataFrame(df.cleanedReview.tolist(), index=df.index)

    countVector = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), token_pattern=r'\w{1,}')
    countVector.fit(df["user_reviews"])
    tfidf = countVector.transform(df["user_reviews"])

    model = joblib.load("sentiment_model.pkl")
    result = model.predict(tfidf)
    df['prediction'] = result

    return df


def RecommendationSystem(df):
    df['avg_ratings'] = df.groupby(['id', 'reviews_username'])['reviews_rating'].transform('mean')
    df['avg_ratings'] = df['avg_ratings'].round(2)
    ratings = df.drop_duplicates(subset={"reviews_username", "id"}, keep="first")
    ratings = ratings.dropna(subset=['reviews_username'])

    train, test = train_test_split(ratings, test_size=0.30, random_state=31)
    df_pivot = train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    ).fillna(0)

    dummy_train = train.copy()
    dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x >= 1 else 1)
    dummy_train = dummy_train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    ).fillna(1)

    # Creating the User Similarity Matrix using pairwise_distance function.
    user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0
    # Create a user-product matrix.
    df_pivot = train.pivot(
        index='reviews_username',
        columns='name',
        values='reviews_rating'
    )
    mean = np.nanmean(df_pivot, axis=1)
    df_subtracted = (df_pivot.T - mean).T
    user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0

    user_correlation[user_correlation < 0] = 0
    user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
    user_final_rating = np.multiply(user_predicted_ratings, dummy_train)
    finalratingpath = "./Data/user_final_rating.csv"
    user_final_rating.to_csv(finalratingpath)


