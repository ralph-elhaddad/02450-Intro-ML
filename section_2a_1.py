#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:28:34 2021

@author: miquel
"""
import numpy as np
import pandas as pd

#Obtention of the matrix X
filename = '../Data/heart_failure.csv'
df = pd.read_csv(filename)
raw_data = df.values  
cols = 0,2,4,6,8,11 
Xnostd = raw_data[:, cols]
y=df['serum_creatinine'].values
N, M = Xnostd.shape

#obtention of a standardized dataset:
X = Xnostd - np.ones((N, 1))*Xnostd.mean(0)
X = X*(1/np.std(X,0))
attributeNames=["age", "creatinine_phosphokinase", "ejection_fraction", "platelets", "serum_sodium", "time"]


from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.linspace(0.0001,10000,1000)

# Initialize variables
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,len(lambdas)))
Error_test_rlr = np.empty((K,len(lambdas)))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))
var=0
k=0

for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]  
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas)
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    

    for l in range (0,len(lambdas)):
         # Estimate weights for the optimal value of lambda, on entire training set
         lambdaI = opt_lambda * np.eye(M)
         lambdaI[0,0] = 0 # Do no regularize the bias term
         w_rlr[:,var] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
         # Compute mean squared error with regularization with optimal lambda
         Error_train_rlr[var,l] = np.square(y_train-X_train @ w_rlr[:,var]).sum(axis=0)/y_train.shape[0]
         Error_test_rlr[var,l] = np.square(y_test-X_test @ w_rlr[:,var]).sum(axis=0)/y_test.shape[0]
    var=var+1
    
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
        # To inspect the used indices, use these print statements
        #print('Cross validation fold {0}/{1}:'.format(k+1,K))
        #print('Train indices: {0}'.format(train_index))
        #print('Test indices: {0}\n'.format(test_index))
        
    k+=1


show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
