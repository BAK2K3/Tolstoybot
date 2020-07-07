# Tolstoybot

Tolstoybot is a deep experimental stateful Recurrent Neural Network for text generation, in the style of Tolstoy! Please visit <https://baknet.herokuapp.com/rnn/tolstoybot> for an online demonstration of this model. 

## Installation

TensorFlow and Numpy are required. To install the required dependencies to your current environment, run:

```bash
$ pip install -r requirements.txt
```

## tolstoybot_train.py

This script will read the War and Peace (Tolstoy) text provided, perform all the required mappings, create a model for training, train the model, then save a version of the model which can be used for testing. The arguments required for this script are as follows:

```python
python tolstoybot_train.py GPU or CPU
```

Please specify when running the script whether you intend to train the model on GPU or CPU. Please be aware that training time varies vastly depending on this, with CPU training time being significantly longer.

This file will output tolstoybot_train.h5 and tolstoybot_test.h5, saved models from TensorFlow.

## tolstoybot_test.py

This script allows a user to test the model, with a given input string, amount of characters to generate, and the "temperature" of the probabilities. 

Temperature scales from 1% to 200%, with the lower percentage generating much more predictable text.  

The amount of characters to generate (num_generate) has been limited to 2000 for the purposes of demonstration, however this can be changed manually from within the script.  

The arguments required for this script are as follows:

```python
python tolstoybot_test.py "start_seed" "num_generate" "temperature"
```
For example:
```python
python tolstoybot_test.py "This is a test" 500 100
```
Produces output as such:

This is a testain pretending to ask him to say that he had reached the
Emperor who was the soldier of the people with the movement of the conception of
the country. The last days were being led him and go to the cause of the
army, and the count’s hussar and such full of strength and considerations and
the people are essential to the Emperor and have been in the same man days in the
conception of the character in the army because they have been
received and she went out of the way had been in the commander in


## tolstoybot.ipynb

This is a jupyter-notebook for step by step execution of training and testing the model. There are no limitations on the temperature and num_generate in this notebook.

## Example Models

Example test and train .h5 files have been provided. These have been trained on a GPU and therefore may require GPU enabled TensorFlow to utilise.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Contact
Please contact me at benjamin.a.kavanagh@gmail.com for any queries. 