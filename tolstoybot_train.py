#imports
import os
#Removed non-critical logs from TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import argparse
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.losses import sparse_categorical_crossentropy

#Checks to see whether either training or testing file exists
training_file = os.path.isfile('tolstoybot_train.h5')
testing_file = os.path.isfile('tolstoybot_test.h5')
if training_file or testing_file:
    print("Tolstoybot training or testing file already exists! Please move or delete this file, then try again")
    sys.exit(1)
else:
    pass

#Parse arguments for GPU or CPU Training
parser = argparse.ArgumentParser()
parser.add_argument(dest='tf_method', help="CPU or GPU Training", choices=['CPU', 'GPU'])
args = parser.parse_args()
print(f"{args.tf_method} selected!")

#Select GPU if selected
if args.tf_method=='GPU':
    tf.compat.v1.reset_default_graph()
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

#Open tolstoybot.txt
path_to_file = "tolstoybot.txt"
text = open(path_to_file, 'r', encoding='utf-8').read()
print("tolstoybot.txt located, and read successuly!")

#Obtain unique vocab set and size 
vocab = sorted(set(text))
vocab_size = len(vocab)

#Create an index mapping for characters in vocab (bi-directional)
char_to_ind = {char:ind for ind, char in enumerate(vocab)}
ind_to_char = np.array(vocab)

print("Vocabulatary obtained, and index mapping successful!")

#Encode whole of text file using vocab index map
encoded_text = np.array([char_to_ind[c] for c in text])

#Set length of sequence
seq_len = 128

#Determine total number of sequences in text
total_num_seq = len(text) // (seq_len+1)

#Create a Character Dataset from encoded text 
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
print("Text encoded, dataset created, and sequences split!")

#Combine into sequences
sequences = char_dataset.batch(seq_len+1, drop_remainder=True)

# Create method for splitting sequences into input/target 
def create_seq_targets(seq):
    input_txt = seq[:-1]
    target_txt = seq[1:]
    return input_txt, target_txt

#Use method to create complete training dataset
dataset = sequences.map(create_seq_targets)

print("Dataset sequenced, split and inititialised for training!")

#Set batch size
batch_size = 32
#Set buffer size
buffer_size = 10000
#Shuffle Dataset
dataset = dataset.shuffle(buffer_size).batch(batch_size,drop_remainder=True)

print("Dataset Shuffled!")

#Set embedding dimensions
embed_dim = 256
#Choose amount of neurons in GRU layers
rnn_neurons4 = 1024
rnn_neurons3 = 512
rnn_neurons2 = 256
rnn_neurons = 105
#Set epochs 
epochs = 100

#Customised Sparse_cat_cross loss function 
def sparse_cat_loss(y_true,y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

#Early Stopping function 
early_stop = EarlyStopping(monitor='loss', patience=3)

#Function for creating model
def create_model(vocab_size, embed_dim,rnn_neurons,rnn_neurons2,rnn_neurons3,rnn_neurons4,batch_size):
    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim, batch_input_shape=[batch_size,None]))
    model.add(SpatialDropout1D(0.2))
 
    model.add(GRU(rnn_neurons, return_sequences=True, 
                 stateful=True, recurrent_initializer='glorot_uniform', batch_input_shape=[batch_size,None, embed_dim]))

    model.add(Dropout(0.2))
    
    model.add(GRU(rnn_neurons2, return_sequences=True, 
                 stateful=True, recurrent_initializer='glorot_uniform'))

    model.add(Dropout(0.2))
    
    model.add(GRU(rnn_neurons3, return_sequences=True, 
                 stateful=True, recurrent_initializer='glorot_uniform'))

    model.add(Dropout(0.2))
    
    model.add(GRU(rnn_neurons4, return_sequences=True, 
                 stateful=True, recurrent_initializer='glorot_uniform'))
    
    model.add(Dropout(0.2))
    
    model.add(Dense(vocab_size))
    model.compile('adam', loss=sparse_cat_loss, metrics=['accuracy'])
    return model

# Create training model
model = create_model(vocab_size=vocab_size,
                     embed_dim=embed_dim,
                     rnn_neurons=rnn_neurons,
                     rnn_neurons2=rnn_neurons2,
                     rnn_neurons3=rnn_neurons3,
                     rnn_neurons4=rnn_neurons4,
                     batch_size=batch_size)

print("Training Model created!")

#Fit the model
print("Training model...")
model.fit(dataset,epochs=epochs,callbacks=[early_stop])
print('Training Successful!')

#Save the model
model.save('tolstoybot_train.h5')
print('Training model saved!')

#Create a new version of the model (test_model) with a single input batch size
test_model = create_model(vocab_size,embed_dim,rnn_neurons, rnn_neurons2, rnn_neurons3, rnn_neurons4, batch_size=1)
test_model.load_weights('tolstoybot_train.h5')
test_model.build(tf.TensorShape([1,None]))

#Save the test_model
test_model.save('tolstoybot_test.h5')
print('Testing model saved!')

#Confirmation
print('Please run tolstoybot_test.py to test the model!')
sys.exit(0)
