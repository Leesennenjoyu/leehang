"""

"""
import numpy as np
import time

def loadData(fileName):
    """
    加载文件
    :param fileName: 路径
    :return: 数据集和数据标签
    """
    print('start read file!')

    # 存放数据及标记
    dataArr = []; labelArr = []
    # 读取文件
    fr = open(fileName)
    # 遍历文件的每一行
    for line in fr.readlines():
        # 获取当前行，并按"，"切割成字段放入列表
        # strip: 去掉每行字符串首尾指定的字符（默认空格或换行符）
        # split: 按照指定的字符将字符串切割，返回列表
        curline = line.strip().split(',')
        dataArr.append([int(num) for num in curline[1: ]])
        labelArr.append(int(curline[0]))
    return dataArr, labelArr

def calcDist(x1, x2):
    """
    计算两个样本点向量之间的距离
    欧氏距离
    :param x1: 向量1
    :param x2: 向量2
    :return: 向量之间的欧式距离
    """
    return np.sqrt(np.sum(np.square(x1 - x2)))
    # 马哈顿距离计算公式
    # return np.abs(np.sum(x1 - x2))

def getClosest(trainDataMat, trainLabelMat, x, topK):
    """
    预测样本x的标记
    获取方式通过找到与样本x最近的topK个点，并查看他们的标签。
    查找里面占某类标签最多的那类标签
    :param trainDataMat: 训练集
    :param trainLabelMat: 训练集标签
    :param x: 要预测的样本
    :param topK: 选择参考最邻近样本的数目
    :return: 预测的标记
    """
    # 建立一个存放向量x与每个训练集中样本距离的列表
    # 列表的长度为训练集的长度，dataList[i]表示x与训练集中第i个样本的距离
    distList = [0] * len(trainLabelMat)

    for i in range(len(trainDataMat)):
        x1 = trainDataMat[i]
        curDist = calcDist(x1, x)
        distList[i] = curDist

    topKList = np.agsort(np.array(distList))[:topK]

    labelList = [0] * 10

    for index in topKList:

        labelList[int(trainLabelMat[index])] += 1

    return labelList.index(max(labelList))

def model_test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, topK):

    print('start test')
    trainDataMat = np.mat(trainDataArr); trainLabelMat = np.mat(trainLabelArr).T
    testDataMat = np.mat(testDataArr); testLabelMat = np.mat(testLabelArr).T

    errorCnt = 0
    # for i in range(len(testDataMat)):
    for i in range(200):
        print('test %d:%d'%(i, 200))
    #     print('test %d:%d' % (i, len(trainDataArr)))
        x = testDataMat[i]
        y = getClosest(trainDataMat, trainLabelMat, x, topK)

        if y != testLabelMat[i]: errorCnt += 1

    return 1 - (errorCnt / 200)
    # return 1 - (errorCnt / len(testDataMat))

if __name__ == "__main__":
    start = time.time()
    trainDataArr, trainLabelArr = loadData('/Users/leesennenjoyu/Words_Words/dataset/mnist_train.csv')
    testDataArr, testLabelArr = loadData("/Users/leesennenjoyu/Words_Words/dataset/mnist_test.csv")
    accur = model_test(trainDataArr, trainLabelArr, testDataArr, testLabelArr, 25)
    print('accur is: %d'%(accur * 100), "%")
    end = time.time()
    print(end - start)
