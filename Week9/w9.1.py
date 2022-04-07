
import pickle

import numpy as np

model = pickle.load(open('W9Model.pkl', 'rb'))  # loads the saved model data

model_params = model.get_params()  # gets the parametres of the mlp

print(model.t_)

print(model.n_outputs_)  # prints out the output 3 outputs but rm there is 2 hidden which includes input and output

print(len(model_params['hidden_layer_sizes']))  # prints out the length of the hidden layer sizes

print(model_params['solver'])  # prints out what solver was used

print(np.shape(model.coefs_[0]))  # gets the shape for the first hidden layer

print(np.shape(model.coefs_[1]))  # does the same for the second

print(np.shape(model.coefs_[3]))  # and does that for the third

print(model_params['activation'])  # prints out the activation type of the model

print(np.shape(model.coefs_[-1][-1]))  # prints out the shape of the model

print(model_params['hidden_layer_sizes'][2])  # gets layer size of the second perceptron