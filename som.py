# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import minisom    #minisom is a general open source

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')  #creating a data frame by reading data from a 'comma separated file'
X = dataset.iloc[:, :-1].values  #selecting all the dependent variable
y = dataset.iloc[:, -1].values  #selecting the dependent variable

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))     #scaling the values of all the attributes to values between 0 and 1
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
dim=10        #initializing the dimensions of the self organizing map ie 10x10
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

#Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone() #creates a blank window to display the results on
pcolor(som.distance_map().T) 
colorbar()  #draws a bar representing a specific color wrt the magnitude of the mean inter-neuron distances(between 0 and 1) obtained after training the SOM 
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):   #i is the index for each customer and x is the vector of customer at every iteration
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markeredgecolor = colors[y[i]], markerfacecolor = 'None', markersize = 10, markeredgewidth = 2)
show()

#finding out the actual persons who are likely to commit fraud
mappings = som.win_map(X)
potential_fraudsters = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis=0)    #(8,1) and (6,8) are the co-ordinates of the blocks denoting the outliers
#doing an inverse transform to get back the original values as we had used minmax scaler before the training process
potential_fraudsters = sc.inverse_transform(potential_fraudsters)

