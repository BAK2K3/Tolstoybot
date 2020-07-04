#imports
import os
#Removed non-critical logs from TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import argparse
import sys

#Parse arguments text generation
parser = argparse.ArgumentParser()
parser.add_argument(dest='start_seed', help="Text to begin generation from!", type=str, default="test")
parser.add_argument(dest='num_generate', help="Amount of characters to generate! (Between 1 and 2000)", type=int, default=500)
parser.add_argument(dest='temperature', help="Temperature of text! (between 1 and 200)", type=int, default=100)
args = parser.parse_args()
#Set the arguments to the required variables
start_seed = args.start_seed
num_generate = args.num_generate
temperature = args.temperature/100

#Check Arguments
if num_generate < 1 or num_generate > 2000:
    print("Please make sure your 'num_generate' argument is between 1 and 1000!")
    sys.exit(1)

if temperature < 0.01 or temperature > 2:
    print("Please make sure your 'temperature' argument is between 1 and 200!")
    sys.exit(1)

print(f"Generating {args.num_generate} characters, with {args.temperature}% temperature, starting with string: {args.start_seed}!")

#only import np and tf if correct arguments are given
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

print("...")

#Open tolstoybot.txt
path_to_file = "tolstoybot.txt"
text = open(path_to_file, 'r', encoding='utf-8').read()
print("Tolstoybot.txt located, and read successuly!")

#Obtain unique vocab set and size 
vocab = sorted(set(text))
vocab_size = len(vocab)

#Create an index mapping for characters in vocab (bi-directional)
char_to_ind = {char:ind for ind, char in enumerate(vocab)}
ind_to_char = np.array(vocab)

print("Vocabulatary obtained, and index mapping successful!")

#Load the model to memory
model = load_model('tolstoybot_test.h5', compile=False)

#Test function for generating consecutive text
def generate_text(model,start_seed,num_generate=500,temperature=1):
    
    #Map each character in start_seed to it's relative index
    input_eval = [char_to_ind[s] for s in start_seed]
    
    #Expand the dimensions
    input_eval = tf.expand_dims(input_eval, 0)
    
    #Create empty list for generated text
    text_generated = []
    
    #Reset the states of tyhe model        
    model.reset_states()
    
    #for each iteration of num_generate
    for i in range(num_generate):
        
        #Obtain probability matrix for current iteration 
        predictions = model(input_eval)
        
        #Reduce dimensions
        predictions = tf.squeeze(predictions,0)
        
        #Multiply probabily matrix by temperature
        predictions = predictions/temperature
        
        #Select a random outcome, based on the unnormalised log-probabilities produced by the model
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
        #Expand dimensions of prediction and assign as next input evaluation
        input_eval = tf.expand_dims([predicted_id], 0)
        
        #Convert prediction to char and append to generated list of text
        text_generated.append(ind_to_char[predicted_id])
        
    #return the initial input, concatenated with the generated text. 
    return(start_seed+"".join(text_generated))

print("...")
print(generate_text(model, start_seed, num_generate, temperature))