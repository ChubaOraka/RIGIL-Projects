
# LSTM with Dropout for sequence classification in the IMDB dataset
import numpy
import pandas as pd
import numpy as np
import nltk

from nltk.corpus import stopwords
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)

#STEP 1:  DATA PREPROCESSING
# load the dataset but only keep the top n words, zero the rest
top_words = 1000
file = pd.read_csv('sample.csv')

#Test data to be assigned categories
x_test_to_category = file['description']

#Train_Data
data = pd.read_csv('data.csv')
x_train = data['description']
y_train = data['name']
#MODIFY Y LABEL DATA TO GET JUST THE FIRST TWO CHARACTERS FOR THE CATEGORIES
# 16 NAMES IN TOTAL TO CLASSIFY DESCRIPTION 
data['Label'] = data['name'].str[:2]
y_train = data['Label']

#Encoding categorical data ( Y Labels to categories)
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
y_train[:] = labelencoder_Y.fit_transform(y_train[:])
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(y_train)




#STEP 2: PROCESS TEXT DATA INTO VECTORS FOR NEURAL NETWORKS
# truncate and pad input sequences

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from keras.utils import to_categorical

max_review_length = 200
output_classes = 18
embedding_vecor_length = 32

#num_words is tne number of unique words in the sequence, if there's more top count words are taken
tokenizer = Tokenizer(top_words)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)
word_index = tokenizer.word_index
input_dim = len(word_index) + 1
print('Found %s unique tokens.' % len(word_index))

#Splitting dataset into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(sequences, dummy_y, test_size = 0.1, random_state = 0)

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model

model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(output_dim = output_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, epochs=60, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#PART 4: TESTING AND ASSIGN LABELS TO UNASSIGNED DESCRIPTION
#PREPROCESS THE TEST DATA
test_sequences = tokenizer.texts_to_sequences(x_test_to_category)
X_test_to_category = sequence.pad_sequences(test_sequences, maxlen=max_review_length)

#START PREDICTING
labels = model.predict(X_test_to_category)
labels = [np.argmax(label, axis=None, out=None) for label in labels]
labels = pd.DataFrame(list(labelencoder_Y.inverse_transform(labels)))
result = pd.concat([x_test_to_category, labels], axis = 1)

writer = pd.ExcelWriter('LSTM.xlsx')

result.to_excel(writer, 'Sheet1')