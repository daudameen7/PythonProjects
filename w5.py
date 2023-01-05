import pandas
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, plot_confusion_matrix
import pickle

test = 'iot_devices_test.csv'   # creates variables w the csv files
train = 'iot_devices_train.csv'
testData = pandas.read_csv(test)  # uses pandas to read the csv file
trainData = pandas.read_csv(train)

npTrain = np.array(trainData)  # creates np arrays and converts the data to the array
npTest = np.array(testData)

trainLabels = npTrain[:, -1]  # this extracts all the data
trainFeatures = npTrain[:, 0:-1] # extracts all the features
testLabels = npTest[:, -1]
testFeatures = npTest[:, 0:-1]

trainFeatures = ((trainFeatures - np.min(trainFeatures)) / (np.max(trainFeatures) - np.min(trainFeatures)))
testFeatures = ((testFeatures - np.min(testFeatures)) / (np.max(testFeatures) - np.min(testFeatures)))
# the code above normalises data

myEncoder = LabelEncoder()  # creates an encoder

trainLabels = myEncoder.fit_transform(trainLabels)  # fits the labels tp the encoder
testLabels = myEncoder.transform(testLabels)

model = DecisionTreeClassifier()  # makes the model a decision tree
model.fit(trainFeatures, trainLabels)  # fits the model using the test and train labels


predict = model.predict(trainFeatures) # makes a prediction
accuracy = accuracy_score(trainLabels, predict) # creates the accuracy score

print(accuracy*100)  # gets the accuracy of the model into percentage


trainScore = model.score(trainFeatures, trainLabels)  # gets the train score using features and labels
print(trainScore*100)  # prints the score into percentage

PrecisionScore = precision_score(trainLabels, predict, average='weighted')  # creates a variable that gets the average
print(PrecisionScore *100)  # prints the precision score into percentage

recallScore = recall_score(trainLabels, predict, average='weighted')  # creates a variable called recall
print(recallScore*100)  # prints the recall score into percentage

f1 = (2*(PrecisionScore*recallScore) / (PrecisionScore+recallScore))  # gets the f1
print(f1)  # prints the f1

classReport = classification_report(trainLabels, predict)
print(classReport)  # prints the class report of the program

file1 = open('TrainedModel.pkl', mode='wb')
pickle.dump(model, file1)
file1.close() # creates a pickle file and saves it to the disk

