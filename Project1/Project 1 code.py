# exercise 1.5.1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Load the Iris csv data using the Pandas library
filename = '../Data/heart_failure.csv'
df = pd.read_csv(filename)

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
raw_data = df.values  

# Notice that raw_data both contains the information we want to store in an array
# X (the sepal and petal dimensions) and the information that we wish to store 
# in y (the class labels, that is the iris species).

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the four columns from inspecting 
# the file.
cols = 0,2,4,6,7,8,11 
X = raw_data[:, cols]

# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape

# Subtract the mean from the data and divide by the attribute standard
# deviation to obtain a standardized dataset:
Y = X - np.ones((N, 1))*X.mean(0)
Y = Y*(1/np.std(Y,0))

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()
