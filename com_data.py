import os
import numpy as np 
import matplotlib.pyplot as plt 
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
cwd = os.path.dirname(os.path.abspath(__file__))+'\\'

num_records=200

def read(folder):
    path = cwd + folder + '\\'
    p = [f'FN{i}acc.txt' for i in range(1,num_records+1)]
    files = np.asarray([np.loadtxt(path+i,dtype=np.float64) for i in p])
    return files

def read_run(folder):
    path = cwd + folder + '\\'
    p = [f'IdaJBSD{i}.txt' for i in range(1,num_records+1)]
    files = np.asarray([np.loadtxt(path+i,dtype=np.float64) for i in p])
    return files

path = cwd + 'OriginalRecords_unf' + '\\'


def readtxt(fname):
    files = np.loadtxt(cwd+fname)
    return files


records = read(folder = 'OriginalRecords_unf')
runs = read_run(folder='runs')

# print(records.shape)
# print(runs.shape)

ac = readtxt(fname='IMs.txt')
re = readtxt(fname='SaAtFundamentalPeriod.txt')
scale_factors = np.outer(1/re[:,1],ac)

# print(scale_factors.shape)
# print(runs.shape)

outputs = []
row_param = len(ac)
treshhold = 1.4e-2

for index in range(len(runs)):
    output = runs[index][:,1]
    output[output < treshhold] = 0
    output[output > treshhold] = 1

    output2 = output

    if len(output) < row_param:
        output = np.append(arr=output, values=np.ones(row_param-len(output)))
    
    # print(output2)
    # print(output,'\n\n\n')

    # print(runs[index][:,3])
    # print(scale_factors[index].shape)

    outputs.append(output)

outputs = np.asarray(outputs)
time = records[0,:,0]
accelerations = np.asarray([np.outer(records[i,:,1], scale_factors[i]).T for i in range(scale_factors.shape[0])])

print('time:', time.shape, '\nouputs:', outputs.shape, '\naccelerations:', accelerations.shape)

np.save(cwd+'time.npy',time)
np.save(cwd+'outputs.npy',outputs)
np.save(cwd+'accelerations.npy',accelerations)
