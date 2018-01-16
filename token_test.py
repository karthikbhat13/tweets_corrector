from random import randint
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers import *
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import utils
from keras.models import *
import keras.backend as K
import numpy as np


text = ["this is the first sentence","this is the second"]
#text = ["this is the first"]

MAX_Length = 50
tokenizer = Tokenizer(num_words = 1000)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
word_index = tokenizer.word_index

TweetTokenizer
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen = MAX_Length)

#labels =  utils.to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)
print(data.shape[0])
print(data)