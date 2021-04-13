"""
Dataset:Mnist
---------------------
Results
    acc:84.3%
    time:91s
"""

import numpy as np
import time

def loadData(fileName):

    dataArr = []; labelArr = []
    fr = open(fileName)
    for line in fr.readlines():

        curLine = line.strip().split(',')
        dataArr.append([int(int(num) > 128) for num in curLine[1: ]])
        labelArr.append(int(curLine[0]))

    return dataArr, labelArr

def NaiveBayes(Py, Px_y, x):
    """
    :param Py: 先验概率分布
    :param Px_y: 条件概率分布
    :param x: 要估计的样本
    :return: 返回所有label的估计概率
    """

    featrueNum  = 784 # 28*28
    classNum = 10

    P = [0] * classNum

    for i in range(classNum):

        sum = 0
        for j in range(featrueNum):
            sum += Px_y[i][j][x[j]]

        P[i] = sum + Py[i]

    return P.index(max(P))

def model_test(Py, P_x_y, testDataArr, testLabelArr):

    errorCnt = 0
    for i in range(len(testDataArr)):
        presict = NaiveBayes(Py, Px_y, testDataArr[i])

        if presict != testLabelArr[i]:
            errorCnt += 1

    return 1 - (errorCnt / len(testDataArr))

def getAllProbability(trainDataArr, trainLabelArr):

    featureNum = 784
    classNum = 10

    Py = np.zeros((classNum, 1))

    for i in range(classNum):
        # Reference formula 4.11
        # K 可取值个数
        Py[i] = ((np.sum(np.mat(trainLabelArr) == i)) + 1) / (len(trainLabelArr) + 10)

    Py = np.log(Py)

    Px_y = np.zeros((classNum, featureNum, 2))

    for i in range(len(trainLabelArr)):

        label = trainLabelArr[i]
        x = trainDataArr[i]
        for j in range(featureNum):
            Px_y[label][j][x[j]] += 1

    for label in range(classNum):
        for j in range(featureNum):
            Px_y0 = Px_y[label][j][0]
            Px_y1 = Px_y[label][j][1]
            Px_y[label][j][0] = np.log((Px_y0 + 1) / (Px_y0 + Px_y1 + 2))
            Px_y[label][j][1] = np.log((Px_y1 + 1) / (Px_y0 + Px_y1 + 2))

    return Py, Px_y

if __name__ == '__main__':
    start = time.time()

    print('start read trainSet')
    trainDataArr, trainLabelArr = loadData("/Users/leesennenjoyu/Words_Words/dataset/mnist_train.csv")

    print('start read testSet')
    testDataArr, testLabelArr = loadData("/Users/leesennenjoyu/Words_Words/dataset/mnist_test.csv")

    print('start to train')

    Py, Px_y = getAllProbability(trainDataArr, trainLabelArr)

    print('start to test')
    accuracy = model_test(Py, Px_y, testDataArr, testLabelArr)

    print('Accur acy:',accuracy)
    print('Time;',time.time() - start)




