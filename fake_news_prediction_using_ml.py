# -*- coding: utf-8 -*-
"""Fake News Prediction Using ML.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Yi-7cN0mqRSwptCWjhFFlNTjvcPADUBP

About  the Dataset:
1. id: unique id for a news article
2. title: author of the news article
4. text: the text of the article; could be incomplete
5.label: a label that marks whether the news article is real or fake.

1: Fake News
0: real news

Importing the dependencies
"""

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk
nltk.download('stopwords')

# prinitng the stopwords in english
print(stopwords.words('english'))

"""Data Processing"""

# loading the data set in pandas dataframe
news_dataset = pd.read_csv('/content/train.csv')

news_dataset.shape

# print first 5 rows of the dataframe
news_dataset.head()

# counting the number of missing values in the dataset
news_dataset.isnull().sum()

# replacing the null values with epmty string
news_dataset = news_dataset.fillna('')

# merging the author name and news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

print(news_dataset['content'])

# seperating the data & label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

print(X)
print(Y)

"""Stemming: Stemming is the process of reducing a word to its root word

example : actors acting actor ---> act
"""

port_stem = PorterStemmer()

def stemming(content):
  stemmed_content = re.sub('[^a-zA-Z]',' ',content)
  stemmed_content = stemmed_content.lower()
  stemmed_content = stemmed_content.split()
  stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] # Changed this line from '-' to '=' to assign the result back to stemmed_content
  stemmed_content = ' '.join(stemmed_content)
  return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

print(news_dataset['content'])

#seperating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values

print(X)

print(Y)

Y.shape

# converting the textual data to numerical data
vectorizer = TfidfVectorizer() # Create an instance of the TfidfVectorizer class
vectorizer.fit(X)

X = vectorizer.transform(X)

"""Splitting Dataset to Training and Test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

"""Training the Logistic Regression"""

model = LogisticRegression()

model.fit(X_train, Y_train)

"""Evaluation

accuracy score
"""

# accuracy score on the trining data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy for test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

"""Making a Predictive System"""

X_new = X_test[1]

prediction = model.predict(X_new)
print(prediction)

if(prediction[0]==0):
  print('The news is Real')
else:
  print("The news is Fake")

print(Y_test[1])

