from mpi4py import MPI
from numpy import random
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import time as t
import datetime as d
import sklearn.metrics as mt
comm=MPI.COMM_WORLD
rank=comm.Get_rank()


def lemmatize(pos_data):
    lemma_rew = " "
    wordnet_lemmatizer = WordNetLemmatizer()
    for word, pos in pos_data:
      if not pos:
        lemma = word
        lemma_rew = lemma_rew + " " + lemma
      else:
        lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
        lemma_rew = lemma_rew + " " + lemma
    return lemma_rew
def clean(text):
    pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
    text = re.sub('[^A-Za-z]+', ' ', text)
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist
def sentiment(i):
  if i == 3:
    return 'Neutral'
  elif i<3:
    return 'Negative'
  else:
    return 'Positive'
if rank==0:
    x = MPI.Wtime()
    Reviews=pd.read_csv(r"C:\Users\this pc\Downloads\Reviews (1).csv")
    print(Reviews)
    Reviews['POS tagged'] = Reviews['Review Text'].apply(clean)
    Reviews['Lemma'] = Reviews['POS tagged'].apply(lemmatize)
    Reviews['Sentiment'] = Reviews['Rating'].apply(sentiment)
    vectorizer = CountVectorizer()
    train_data,test_data = train_test_split(Reviews,test_size=0.3)
    X_train = vectorizer.fit_transform(train_data["Review Text"].fillna(' '))
    y_train = train_data['Sentiment']
    X_test = vectorizer.transform(test_data["Review Text"].fillna(' '))
    y_test = test_data['Sentiment']
    start=d.datetime.now()
    nn = MLPClassifier(alpha=0.002,solver='lbfgs',random_state=1,learning_rate_init=0.25)
    nn.fit(X_train, y_train)
    pred = nn.predict(X_test)
    print('Elapsed time in MLP classifier: ',str(d.datetime.now()-start))
    print('Accuracy score given by MLP:',mt.accuracy_score(y_test, pred))
    start=d.datetime.now()
    svm = SVC()
    svm.fit(X_train, y_train)
    pred = svm.predict(X_test)
    print('Elapsed time in svm: ',str(d.datetime.now()-start))
    print('Accuracy score by SVM:',mt.accuracy_score(y_test, pred))
    start=d.datetime.now()
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    pred = neigh.predict(X_test)
    print('Elapsed time in KNN: ',str(d.datetime.now()-start))
    print('Accuracy score by KNN:',mt.accuracy_score(y_test, pred))
    start=d.datetime.now()
    nb= MultinomialNB()
    nb.fit(X_train, y_train)
    pred = nb.predict(X_test)
    print('Elapsed time in Naive Bayes: ',str(d.datetime.now()-start))
    print('Accuracy score in Naive Bayes:',mt.accuracy_score(y_test, pred))
    start=d.datetime.now()
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    pred = dt.predict(X_test)
    print('Elapsed time in Decision Tree: ',str(d.datetime.now()-start))
    print('Accuracy score in Decision Tree :',mt.accuracy_score(y_test, pred))
    y = MPI.Wtime()
    print("Time elapsed: ",y-x)

