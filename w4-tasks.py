
import numpy as np
import pandas
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,classification_report,confusion_matrix

file = 'spam_datasetcopy.csv'  # loads the file

dataset = pandas.read_csv('spam_datasetcopy.csv')

np_array = dataset.to_numpy() # converts the data to a numpy array

labels = np_array[:, -1] # extracts the labels of the data

features = np_array[:,:-1]# extracts the features

print(features) # prints out the features of the data

print(labels) # prints out data labels

print(labels.shape) # gets shape of labels

print(features.shape) # gets shape of features

myEncoder = LabelEncoder() # this creates a varaible to allow encoding of labels

labels = myEncoder.fit_transform(labels) # fits and transforms labels to be encoded

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.1)

# the code above would create a train test split

print(len(train_features))

print("test features size is: ", len(test_features))

print(len(train_labels))

print(len(test_labels))

model = SVC(kernel='linear', probability=True) # this creates a support vector machine which is linear

SVC = model.fit(train_features, train_labels) # fits the model with the train feature and train labels datasets

trainAcc = model.score(train_features, train_labels) # this uses features and labels to determine the score


probability = model.predict_proba(test_features[0:10, :]) # creates a probaility of the model

print(probability)

testLabelPredicts = model.predict(test_features) # makes predictions using test feature dataset

print(testLabelPredicts)

accScore = accuracy_score(test_labels, testLabelPredicts) # gets the accuracy of the model

print(accScore)

PrecisScore = precision_score(test_labels, testLabelPredicts) # gets precision of the model

print(PrecisScore)

Recall_Score = recall_score(test_labels, testLabelPredicts) # determines the recall score

print(Recall_Score)

F1 = f1_score(test_labels, testLabelPredicts) # calculates the f1 score of the model

print(F1)

print(accScore * 100)

ClassRep = classification_report(test_labels, testLabelPredicts) # creates a classification report

print(ClassRep)

file1 = open('Week4Model.pkl', mode='wb')
pickle.dump(model, file1)
file1.close()  # this code writes the model to disk and saves it as a pickle file

ConMatrix = confusion_matrix(test_labels, testLabelPredicts) # create a confusion matrix
print(ConMatrix) # prints it out