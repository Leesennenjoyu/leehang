"""
Mnist
"""

import numpy as np
import pandas as pd

# Load the data
def DataLoader(dataset):

    df_dataset = pd.read_csv(dataset, header = None)

    # datatrain & labeltrain
    data, label = df_dataset.iloc[:, 1:].copy(), df_dataset.iloc[:, :1].copy()

    # Normalization
    data = data/255

    return data, label

# Distance
def calcDist(x1, x2, Manhattan = True):

    if Manhattan:
        return np.abs(np.sum(x1 - x2))
    else:
        return np.sqrt(np.sum(np.square(x1 - x2)))

# Prediction
def Predict(datatrain, labeltrain, datatest, labeltest, k):

    datatrain, labeltrain, = np.mat(datatrain), np.mat(labeltrain).T
    datatest, labeltest = np.mat(datatest), np.mat(labeltest).T

    m1, n1 = np.shape(datatrain)
    m2, n2 = np.shape(datatest)

    accCnt = 0
    for num1 in range(2):
        disdic = {}
        for num2 in range(m1):
            dis = calcDist(datatest[num1], datatrain[num2], Manhattan = False)
            disdic[num2] = dis

        dislist = sorted(disdic.items(), key = lambda item: item[1], reverse = True)
        topKlist = dislist[:k]

        labelindex = {}
        for index,n in topKlist:
            if labeltrain[0, index] in labelindex:
                labelindex[labeltrain[0, index]] += 1
            else:
                labelindex[labeltrain[0, index]] = 1

        label_p = sorted(labelindex.items(), key = lambda item: item[1])[0][0]

        if label_p == labeltest[0, num1]:
            accCnt += 1
    print(1)
    return accCnt/m2

if __name__ == "__main__":

    datatrain, labeltrain = DataLoader('/Users/leesennenjoyu/Words_Words/dataset/mnist_train.csv')
    datatest, labeltest = DataLoader('/Users/leesennenjoyu/Words_Words/dataset/mnist_test.csv')

    acc = Predict(datatrain, labeltrain, datatest, labeltest, 25)
    print(acc)







