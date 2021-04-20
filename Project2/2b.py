#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 21:28:29 2021

@author: apolloseeds
"""

from dataset import *

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary
import torch
from scipy import stats
from toolbox_02450 import feature_selector_lr, bmplot, rlr_validate, mcnemar
from toolbox_02450 import train_neural_net, draw_neural_net, visualize_decision_boundary


N2, M2 = contX.shape


def trainANN(X,y,h, K = 10): #returns the optimal h (number of hidden units)
    CV = model_selection.KFold(K,shuffle=True)
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000
    
    # Define the model structure
    
    # The lambda-syntax defines an anonymous function, which is used here to 
    # make it easy to make new networks within each cross validation fold
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M2, h), #M features to H hiden units
                        # 1st transfer function, either Tanh or ReLU:
                        torch.nn.Tanh(),                            #torch.nn.ReLU(),
                        torch.nn.Linear(h, 1) # H hidden units to 1 output neuron
                        )
    
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss


    print('Training model of type:\n\n{}\n'.format(str(model())))
    errors = [] # make a list for storing generalizaition error in each loop
    
    for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
        print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K)) 
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index,:])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index,:])
        y_test = torch.Tensor(y[test_index])
        
        
        for i in range(0, len(h)):
            #Iterate over every h
            testedH = h[i]
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            
            print('\n\tBest loss: {}\n'.format(final_loss))
            
            # Determine estimated class labels for test set
            y_sigmoid = net(X_test)
            y_test_est = (y_sigmoid>.5).type(dtype=torch.uint8)
        
            # Determine errors and errors
            y_test = y_test.type(dtype=torch.uint8)
        
            e = y_test_est != y_test
            error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
            errors.append(error_rate) # store error rate for current CV fold 
            
        optimalHIndex = errors.index(min(errors))
        optimalH = h[optimalHIndex]
        
        # Print the average classification error rate
        print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))
        
        return optimalH
        
    
def annRegression(X_train, X_test, y_train, y_test, hRange, K = 10):
    # Parameters for neural network classifier
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 10000         # stop criterion 2 (max epochs in training)
    loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

    opt_hidden_unit = trainANN(X_train, y_train, hRange, K)
    model = lambda: torch.nn.Sequential(
                        torch.nn.Linear(M, opt_hidden_unit), #M features to H hiden units
                        torch.nn.Tanh(),   # 1st transfer function,
                        torch.nn.Linear(opt_hidden_unit, 1), # H hidden units to 1 output neuron
                        )
    # print('Training model of type:\n\n{}\n'.format(str(model())))

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                    loss_fn,
                                                    X=X_train,
                                                    y=y_train,
                                                    n_replicates=n_replicates,
                                                    max_iter=max_iter)
    
    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    
    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    return opt_hidden_unit, mse, y_test_est
    

C = 2
# Normalize data
annX = stats.zscore(contX)


# Parameters for neural network classifier
h = 1 # number of hidden units, !!!!SELECT A RANGE BY TESTING

serumC = np.array(np.asarray(X[:, 7]), dtype=int)
#y_rings = np.array(np.asarray(rings), dtype=np.int).reshape(-1, 1)

K = 5
lambdas = np.linspace(0.01, 10, 1000)
inner_cvf = 10
CV = model_selection.KFold(K, shuffle=True)
coefficient_norm = np.zeros(K)
# Parameters for neural network classifier
hRange = range(1, 8)
n_replicates = 2        # number of networks trained in each k-fold
max_iter = 10000         # stop criterion 2 (max epochs in training)

square_err_regression_base = np.empty(K)
square_err_regression_RLR = np.empty(K)
square_err_regression_ANN = np.empty(K)
regression_RLR_opt_lambdas = np.empty(K)
regression_opt_hidden_units = np.empty(K)

error_rate_classification_base = np.empty(K)
error_rate_classification_logistic = np.empty(K)
error_rate_classification_ANN = np.empty(K)
classification_opt_hidden_units = np.empty(K)
classification_opt_lambdas = np.empty(K)
w_est_logistic_arr = np.empty((K, X.shape[1]))

y_est_Reg_ANN = []
y_est_Reg_RLR = []
y_est_claf_ANN = []
y_est_claf_logistic = []
y_sex_real = []
y_rings_real = []

for k, (train_index, test_index) in enumerate(CV.split(annX,serumC)):
        X_train = annX[train_index,:]
        X_test = annX[test_index,:]
        y_train = serumC[train_index]
        y_test = serumC[test_index]
        """
        y_rings_train = y_rings[train_index]
        y_rings_test = y_rings[test_index]
        y_sex_real.append(y_sex_test)
        y_rings_real.append(y_rings_test)
        """
        

        regression_opt_hidden_unit, ANN_mse, y_est_ANN_regression =  annRegression(X_train, X_test, y_train, y_test, hRange, inner_cvf)
        regression_opt_hidden_units[k] = regression_opt_hidden_unit
        square_err_regression_ANN[k] = ANN_mse
        y_est_Reg_ANN.append(y_est_ANN_regression)
        
    

print("square_err_regression_ANN: ", square_err_regression_ANN)


