import numpy as np
import pandas
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

RawData = 'phishing_dataset.csv'  # loads the data

data = pandas.read_csv(RawData)  # uses pandas to read the data which is a csv file

print(data)  # prints out the pandas data

data1 = np.array(data)  # converts pandas data to numpy array

labels = data1[:, -1]  # extraction of labels

features = data1[:, 0: -1]  # extraction of features

train_feature, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)

# code above is the creation of a training test split which will be used when training the model

model = MLPClassifier(hidden_layer_sizes=(50, 50, 50), early_stopping=True, verbose=True, validation_fraction=0.1,
                      solver='sgd',
                      activation='relu', random_state=0, learning_rate_init=1e-7)

# the code above is the creation of the multi layer perceptron

model.fit(train_feature, train_labels)  # fits the model using both the training features and train label datasets

prediction = model.predict(test_features)  # creates a prediction of the model using the test features data

TrainScore = model.score(train_feature, train_labels)  # creates the training score of the model using train features
# and train labels

accuracy = accuracy_score(test_labels, prediction)  # determines the accuracy of the model

precision_Score = precision_score(test_labels, prediction)  # gets precision score of the model

recallScore = recall_score(test_labels, prediction)  # gets the recall score of the model

confusion_Matrix = confusion_matrix(test_labels, prediction)  # creates the confusion matrix

Class_Rep = classification_report(test_labels, prediction)  # gets classification report of the model

f1__Score = f1_score(test_labels, prediction)  # creates the f1 score for model

plt.plot(model.loss_curve_)  # this plots a graph which shows the training loss of the model
plt.plot(model.validation_scores_) # this would plot the validation score of the model
plt.xlabel('number of training iterations') # makes an x label w that heading
plt.ylabel('loss/score')  # makes a y label
plt.legend(['training-loss', 'validation-score'])  # plots a legend
plt.grid()  # plots the grid
plt.show()  # this would display the graph

