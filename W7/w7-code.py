import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pickle

RawData = 'Mickymouse.jpg'  # assigning the file name
data = plt.imread(RawData)  # loading the data
data2 = data.reshape(-1, 3)  # reshape the image numpy array

model = KMeans(n_clusters=5, random_state=0)  # creating the model w its parameter
model.fit(data2)  # fitting the model w 2d array
predict = model.predict(data2)  # gets models predictions
predict = predict.reshape((data.shape[0], data.shape[1]))  # reshapes predictions to orig data
plt.imshow(predict, cmap='gray')  # views the clusters labels
loc = np.where(predict == 0)  # finds the pixels that belongs to 0 cluster
newImage = data.copy()  # creates copy of data
newImage[loc[0], loc[1], :] = [255, 100, 1]  # edits the image using rgb colours 0-255

plt.imshow(data)  # displays original image
plt.show()
plt.figure()  # creates new figure
plt.imshow(newImage)  # shows the edited image
plt.show()

# creates a pickle model
file1 = open('Week7Model.pkl', mode='wb')
pickle.dump(model, file1)
file1.close()
