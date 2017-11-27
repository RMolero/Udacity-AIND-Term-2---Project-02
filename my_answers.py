import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for i in range(len(series)-window_size):
        X.append(series[i:i + window_size])
    
    y = series[window_size:]
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    # model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    
    punctuation = ['!', ',', '.', ':', ';', '?']
    lower_cases = [chr(i) for i in range(97,123)]
    empty_space = [' ']
    
    valid_chars = lower_cases + punctuation + empty_space
    answer = []
    
    for i in range(len(text)):
        if (text[i] in valid_chars):
            answer.append(text[i])
        else:
            answer.append(' ')
            
    answer_str = ''.join(answer)
    return answer_str

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    # As suggested by reviewer
    inputs = [text[s:s+window_size] for s in range(0,len(text)-window_size,step_size)]
    outputs = [text[s] for s in range(window_size,len(text),step_size)]
    
    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    
    model = Sequential()
    model.add(LSTM(200, activation='tanh', input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    
    return model
