%matplotlib inline
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist
import re
import sys
import string
import random
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# from tf.keras.models import Sequential  # This does not work!
# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import load_model

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
#from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args


data = pd.read_excel('Training Data for Primary.xlsx',sheetname = 'Sheet1')
data = data.sample(frac=1) #shuffle data
data_pred = pd.read_excel('Primary to be decided.xlsx',sheetname = 'Sheet1')

###############
punc = string.punctuation
#stemmer = SnowballStemmer('english')
#words = stopwords.words("english")
punc = punc.replace("&", "") # don't remove &
pattern = r"[{}]".format(punc) # create the pattern

data['cleaned'] = data['Business Description'].apply(lambda x: " ".join([i for i in re.sub(pattern, " ", x).split()]).lower())
#data['cleaned'] = data['cleaned'].apply(lambda x: ''.join(ch for ch in x  if ch not in pattern))
data['cleaned'] = data['cleaned'].str.strip()
########################

X = data['cleaned']
y = data['Existing PIC']
from sklearn.preprocessing import LabelBinarizer
le = LabelBinarizer()
y = le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1,stratify=y)

########
num_words = None
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(x_train)
if num_words is None:
    num_words = len(tokenizer.word_index)+1
###########
x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)
#########
num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)
np.mean(num_tokens)
np.max(num_tokens)
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
#########

pad = 'pre'
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)

#########

idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))

def tokens_to_string(tokens):
    # Map from tokens back to words.
    words = [inverse_map[token] for token in tokens if token != 0]
    
    # Concatenate all words.
    text = " ".join(words)

    return text

############
    
# =============================================================================
# from sklearn.utils.class_weight import compute_class_weight
# y_integers = np.argmax(y_train,axis=1)
# class_weights = compute_class_weight('balanced',np.unique(y_integers),y_integers)
# d_class_weights = dict(enumerate(class_weights))  
#model.fit(x_train_pad, y_train,
#          validation_split=0.05, epochs=1, batch_size=64,class_weight=d_class_weights)                            
# =============================================================================

############

dim_learning_rate = Real(low=0.005, high=0.1, prior='log-uniform',
                         name='learning_rate')
dim_num_dense_layers = Integer(low=1, high=2, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=25, high=75, name='num_dense_nodes')
dim_emb_size = Categorical(categories=[5,10,20],
                             name='emb_size')
dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_emb_size]
default_parameters = [0.04,2,44,10]

def log_dir_name(learning_rate, num_dense_layers,
                 num_dense_nodes, emb_size):

    # The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       emb_size)

    return log_dir

######
    
def create_model(learning_rate, num_dense_layers,
                 num_dense_nodes, emb_size):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """
    embedding_size = emb_size
    # Start construction of a Keras Sequential model.
    model = Sequential()

    model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    #embeddings_initializer=word2vec
                    input_length=max_tokens,
                    name='layer_embedding'))
    # Add fully-connected / dense layers.
    # The number of layers is a hyper-parameter we want to optimize.
    for i in range(num_dense_layers):
        # Name of the layer. This is not really necessary
        # because Keras should give them unique names.
        name = 'layer_dense_{0}'.format(i+1)
        if num_dense_layers == 1:
            model.add(Bidirectional(LSTM(units=num_dense_nodes)))
            model.add(Dropout(0.2))
        else:
            if i == 0:
                model.add(Bidirectional(LSTM(units=num_dense_nodes, return_sequences=True)))
                model.add(Dropout(0.2))
            else:
                model.add(Bidirectional(LSTM(units=num_dense_nodes)))
                model.add(Dropout(0.2))
            
    model.add(Dense(9, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])        
    return model        

#######################
    
path_best_model = '19_best_model.keras'

best_accuracy = 0.0

##########

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers,
            num_dense_nodes, emb_size):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of nodes in each dense layer.
    activation:        Activation function for all layers.
    """

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('emb_size:', emb_size)
    print()
    
    # Create the neural network with these hyper-parameters.
    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         emb_size = emb_size)

    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, num_dense_layers,
                           num_dense_nodes, emb_size)
    
    # Create a callback-function for Keras which will be
    # run after each epoch has ended during training.
    # This saves the log-files for TensorBoard.
    # Note that there are complications when histogram_freq=1.
    # It might give strange errors and it also does not properly
    # support Keras data-generators for the validation-set.
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        batch_size=32,
        write_graph=True,
        write_grads=False,
        write_images=False)
   
    # Use Keras to train the model.
    history = model.fit(x_train_pad, y_train,
           validation_split=0.2, epochs=10, 
                         batch_size=500,
                        callbacks=[callback_log])

    # Get the classification accuracy on the validation-set
    # after the last training-epoch.
    accuracy = history.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Save the model if it improves on the best-found performance.
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Save the new model to harddisk.
        model.save(path_best_model)
        
        # Update the classification accuracy.
        best_accuracy = accuracy

    # Delete the Keras model with these hyper-parameters from memory.
    del model
    
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    
    # NOTE: Scikit-optimize does minimization so it tries to
    # find a set of hyper-parameters with the LOWEST fitness-value.
    # Because we are interested in the HIGHEST classification
    # accuracy, we need to negate this number so it can be minimized.
    return -accuracy

#######
#Test Run
#fitness(default_parameters)
########

search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=30,
                            x0=default_parameters)

##########
#optimization progress
plot_convergence(search_result)

##########
#Best Hyper Parameter:
search_result.x
#space = search_result.space
space.point_to_dict(search_result.x)
search_result.fun
sorted(zip(search_result.func_vals, search_result.x_iters))

#############

#Evaluate Best Model
model = load_model(path_best_model)
result = model.evaluate(x=x_test_pad,
                        y=y_test)
for name, value in zip(model.metrics_names, result):
    print(name, value)
    
###############

#predict on new data
    
y_pred = model.predict(x=x_test_pad)
cls_pred = np.argmax(y_pred,axis=1)
cls_true = np.argmax(y_test,axis=1)





