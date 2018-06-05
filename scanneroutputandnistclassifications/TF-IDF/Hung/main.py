# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:42:47 2018

@author: hungn
"""

import pandas as pd
import numpy as np

print('hello' )

file = pd.read_csv('sample.csv')





#Test data
x_test = file['description']

#Train_Data
data = pd.read_csv('data.csv')
x_train = data['description']
y_train = data['name']

#Build the model from sklearn

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(x_train)
X_train_counts.shape

#TFIDF Model
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

data['modified description'] = data['name'].str[:2]
y_train = data['modified description']


