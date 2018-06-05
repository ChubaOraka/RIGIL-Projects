# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:42:47 2018

@author: hungn
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')



print('hello' )

file = pd.read_csv('sample.csv')





#Test data
x_final_validate_result = file['description']

#Train_Data
data = pd.read_csv('data.csv')
x_train = data['description']
y_train = data['name']

#Build the model from sklearn

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X_train_counts = count_vect.fit_transform(x_train )
X_train_counts.shape

X_final_validate_result = count_vect.transform(x_final_validate_result)
X_final_validate_result.shape

#TFIDF Model
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

X_final_validate_tfidf = tfidf_transformer.transform(X_final_validate_result)
X_final_validate_tfidf.shape

#MODIFY Y LABEL DATA
data['modified description'] = data['name'].str[:2]
y_train = data['modified description']

#Encoding categorical data ( Y Labels to categories)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_Y = LabelEncoder()
y_train[:] = labelencoder_Y.fit_transform(y_train[:])

#Splitting dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train_tfidf, y_train, test_size = 0.2, random_state = 0)

#Method 1: Naive Bayes Classfier 
from sklearn.naive_bayes import MultinomialNB
NBclassifier = MultinomialNB().fit(X_train, Y_train)

#Testing Naive Bayes Classifier
predicted = NBclassifier.predict(X_test)
np.mean(predicted == Y_test)

#Validate on Sample.csv file
sample_predicted = NBclassifier.predict(X_final_validate_tfidf)
y_final_validate_result = pd.DataFrame(list(labelencoder_Y.inverse_transform(sample_predicted)))
#result
result = pd.concat([x_final_validate_result, y_final_validate_result], axis = 1)
from sklearn.metrics import confusion_matrix
matrix1 = confusion_matrix(Y_test, predicted)




#METHOD 2: SUPPORT VECTOR MACHINES
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

text_clf_svm = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                     ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                     alpha=1e-3, n_iter=5, random_state=42)),])

_ = text_clf_svm.fit(X_train, Y_train)
predicted_svm = text_clf_svm.predict(X_test)
np.mean(predicted_svm == Y_test)
