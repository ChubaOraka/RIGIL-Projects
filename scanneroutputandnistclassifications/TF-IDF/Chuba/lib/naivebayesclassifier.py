# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:03:57 2018

@authors:   Chuba Oraka
            hungn
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:42:47 2018

@author:    Chuba Oraka
            hungn
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB


#Splitting dataset into training set and test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X_train_tfidf, y_train, test_size = 0.2, random_state = 0)

#Method 1: Naive Bayes Classfier 

class NB():
    X_train = None
    Y_train = None
#    y_final = None
    NBclassifier = None
    predicted = None
    
    def __init__(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train
#        self.y_final = y_final
        print("Using Naive-Bayes method")
        
    def multinomial(self):
        self.NBclassifier = MultinomialNB().fit(self.X_train, self.Y_train)
        
    def test(self, X_test, Y_test):
        print("Running tests")
        predicted = self.NBclassifier.predict(X_test)
        accuracy = np.mean(predicted == Y_test)
        print("You have an accuracy of ", accuracy)
        from sklearn.metrics import confusion_matrix
        matrix1 = confusion_matrix(Y_test, predicted)
        print(matrix1)
        
    def validate(self, x_raw_input, x_trans_input, labelencoder):
        #Validate on Sample.csv file
        sample_predicted = self.NBclassifier.predict(x_trans_input)
        
        y_output = pd.DataFrame(list(labelencoder.inverse_transform(sample_predicted)))
        #result
        result = pd.concat([x_raw_input, y_output], axis = 1)
        #writer = pd.ExcelWriter('TF-IDF, Naive Bayes on Full Sparse Dataset.xlsx')
        
        #result.to_excel(writer, 'Sheet1')


