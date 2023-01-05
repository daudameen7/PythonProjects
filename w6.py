import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

RawData = 'Excelrates-GBP-USD.xlsx'  # reads excel dataset
data = pandas.read_excel(RawData)  # reads dataset in pandas dataframe
npData = np.array(data)  # converts data into numpy array
rate = npData[:, -1]  # extracts features
rate = rate.reshape(-1, 1)  # reshapes into 2d array
print(np.std(rate))
print(np.mean(rate))

# creates an isolation forrest for the model and trains it using the rate
model = IsolationForest(contamination=0.02, bootstrap=True)
model.fit(rate)
predict = model.predict(rate)
print(np.sum(predict))

# finds the total number of anomalies in the data
AbLoc = np.where(predict == -1)[0]
year = np.linspace(2021, 1999, np.shape(rate)[0])
AbNormYears = np.floor(year[AbLoc])
print(np.unique(AbNormYears))

# this does the plotting of the rates and adds the anomalies
plt.plot(year, rate)
plt.plot(year[AbLoc], rate[AbLoc], marker='x', linestyle='')
plt.xlabel('years from 1999 to 2021')
plt.ylabel('gbp-usd ex-rates')
plt.show()