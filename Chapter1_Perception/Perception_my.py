"""
Mnist
"""
import numpy as np
import pandas as pd
import time
from tqdm import trange

# Load the data!
def DataLoader(dataset):

    print('Start to read dataset')
    data_df = pd.read_csv(dataset, header = None)
    # Label & Traindata
    labeltrain, datatrain= data_df.iloc[:,:1].copy(), data_df.iloc[:, 1:].copy()

    # Binarization
    print('Step-1 \nStart binarization!')
    for index in trange(len(labeltrain.index)):
        if labeltrain.iloc[index, 0] > 5:
            labeltrain.iloc[index, 0] = 1
        if labeltrain.iloc[index, 0] <= 5:
            labeltrain.iloc[index, 0] = -1

    # Normalize
    print('Step-2 \nStart normalization!')
    data_df =data_df/255

    print('Finished')
    return labeltrain, datatrain

# Train
def perception(data_label, data_train, iters):

    data_label, data_train = np.mat(data_label).T, np.array(data_train)
    m, n = np.shape(data_train)

    # Initialize
    w = np.zeros((1, np.shape(data_train)[1]))
    b = 0
    step = 0.0001

    # Train
    for iter in trange(iters):
        for i in range(m):
            if -1 * data_label[0, i] * (np.matmul(w, data_train[i].T) + b) >= 0:
                w += step * data_label[0, i] * data_train[i]
                b += step * data_label[0, i]

    return w, b

# Evaluate
def model_evaluate(datatest, labeltest, w, b):

    print('Start to evaluate!')
    datatest, labeltest = np.mat(datatest), np.mat(labeltest).T

    errorCnt = 0

    for i in range(datatest.shape[0]):

        if -1 * labeltest[0, i] * (np.matmul(w, datatest[i].T) + b) >= 0:
            errorCnt += 1
    print(errorCnt)

    return (1 - errorCnt / np.shape(datatest)[0])

if __name__ == "__main__":
    start = time.time()
    labeltrain, datatrain = DataLoader('/Users/leesennenjoyu/Words_Words/dataset/mnist_train.csv')
    labeltest, datatest = DataLoader('/Users/leesennenjoyu/Words_Words/dataset/mnist_test.csv')
    w, b = perception(labeltrain, datatrain, 50)
    print('w is', w)
    print('b is', b)
    acc = model_evaluate(datatest, labeltest, w, b)
    print(acc, time.time()- start)












