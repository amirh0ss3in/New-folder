import numpy as np 
import matplotlib.pyplot as plt 
import warnings
from sklearn.model_selection import train_test_split
import os
import random


def flatten(x):
    X=[]
    for i in x:
        # print(i)
        for j in i:
            X.append(j)
    return np.asarray(X)

np.random.seed(42)

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
cwd = os.path.dirname(os.path.abspath(__file__))+'\\'


time = np.load(cwd+'time.npy')
outputs = np.load(cwd+'outputs.npy')
accelerations = np.load(cwd+'accelerations.npy')

a = np.arange(0,len(accelerations))
test_ind = np.random.choice(a, size=int(0.25 * len(accelerations)), replace=False)
train_ind = np.delete(a, test_ind)

# print(outputs[2])
# print(len(test_ind),len(train_ind))
# print(accelerations[test_ind].shape,accelerations[train_ind].shape)

xtest = flatten(accelerations[test_ind])
xtrain = flatten(accelerations[train_ind])
ytest = flatten(outputs[test_ind])
ytrain = flatten(outputs[train_ind])

xtest = np.reshape(xtest, [xtest.shape[0],xtest.shape[1],1])
xtrain = np.reshape(xtrain, [xtrain.shape[0],xtrain.shape[1],1])

# print(xtest.shape,ytest.shape,xtrain.shape,ytrain.shape)
