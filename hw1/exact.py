import numpy as np
from sklearn.linear_model import LinearRegression

# load csv into pandas dataframe
data = np.genfromtxt('linear.csv', delimiter=',')

# extract feature matrix and labels
features = data[:,1:]
labels = data[:,0]

# init the regressor and call regressor.fit()
regressor = LinearRegression()
regressor.fit(features, labels)

# print results
print(regressor.coef_)



