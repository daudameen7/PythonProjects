import numpy as np
import pandas
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

RawData = 'Spam(1).csv' # loads the raw data

data = pandas.read_csv(RawData)  #uses pandas to read the csv data

NumpyArray = data.to_numpy() # converts to numpy array

label = NumpyArray[:, -1] # etxraction of labels

features = NumpyArray[:, 0:-1] #extraction of features

myEncoder = LabelEncoder() # creates encoder

label = myEncoder.fit_transform(label)  # fits the encoder to label

train_feature, test_features, train_label, test_label = train_test_split(features, label, test_size=0.2) #training test split

model = Perceptron(tol=1e-3, random_state=0) # creates perceptron

model.fit(train_feature, train_label) # fits the model w the train data

trainScore = model.score(train_feature, train_label) # gets training score

prediction = model.predict(test_features) # gets prediction using test feature data

accuracy = accuracy_score(test_label, prediction) # this tests the accuracy score

PrecisScore = precision_score(test_label, prediction) # gets precision score

Recall = recall_score(test_label, prediction)  # gets recall score

F1_Score = f1_score(test_label, prediction) # gets f1 score

Classification_Report = classification_report(test_label, prediction) # creates classification report

print(accuracy * 100) # prints accuracy into percentage

print(PrecisScore * 100) # gets precision score in a percentage

print(Recall * 100) # gets recall score percentage

print(F1_Score * 100) # gets f1 score percentage

print(Classification_Report) # prints out the report

print(np.sum(test_label == prediction)) # gets the sum of the test label and prediction

ConfusionMatrix = confusion_matrix(test_label, prediction) # creates confsuion matrix

print(ConfusionMatrix) # prints it out



file1 = open('Week8Model.pkl', mode='wb')

pickle.dump(model, file1)

file1.close() # creates a pickle file and ensures the mode is write binary and saves the model to disk



