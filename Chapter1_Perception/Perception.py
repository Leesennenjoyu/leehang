"""
数据集：Mnist
"""
import time
import numpy as np

def  loadData(filename):
	
	"""
	加载Mnist数据
	:param
		filename:要加载的数据集路径
	：return
		list形式的数据集及标记
	"""

	print('start to read data')
	# 存放数据及标记的list
	dataArr = []; labelArr = []
	# 打开文件
	fr = open(filename, 'r')
	# 将文件按行读取
	for line in fr.readlines():
		# 对每一行的数据按切割符“，”进行切割，返回字段列表
		curLine = line.strip().split(',')

		# Mnist有0-9，共10个标记，由于是二分类任务，所以将>=5的作为1，<5为-1
		if int(curLine[0]) >= 5:
			labelArr.append(1)
		else:
			labelArr.append(-1)

		# 存放标记
		dataArr.append([int(num)/255 for num in curLine[1:]])

	return dataArr, labelArr


def perception(dataArr, labelArr, iter = 50):
	"""
	感知机训练过程
	：param
		dataArr:训练的数据(list)
		labelArr:训练的标签(list)
		iter: 迭代次数
	"""

	print('start to trans')
	# 将数据转为矩阵形式
	# 转换后的数据每一个样本的向量都是横向的

	dataMat = np.mat(dataArr)
	# 将标签转换为矩阵，之后转置
	# 转置是因为在运算中只需要单独取label中某个元素，如果是1xN的矩阵的话，无法用label[i]的方式读取
	# 对于只有1xN的label可以不转换为矩阵，直接label[i]，这里转换为格式上的统一
	labelMat = np.mat(labelArr).T
	# 获取数据矩阵的大小，为m*n
	m, n = np.shape(dataMat)
	# 创建初始权重w，初始值全为0。
    # np.shape(dataMat)的返回值为m，n -> np.shape(dataMat)[1])的值即为n，与
    # 样本长度保持一致
    w = np.zeros((1, np.shape(dataMat)[1]))
    # 初始化偏置b为0
    b = 0
    # 初始化步长，也就是梯度下降过程中的n，控制梯度下降速率
    h = 0.0001

    # 进行iter次迭代计算
    for k in range(iter):
    	# 对于每一个样本进行梯度下降
        # 李航书中在2.3.1开头部分使用的梯度下降，是全部样本都算一遍以后，统一
        # 进行一次梯度下降
        # 在2.3.1的后半部分可以看到（例如公式2.6 2.7），求和符号没有了，此时用
        # 的是随机梯度下降，即计算一个样本就针对该样本进行一次梯度下降。
        # 两者的差异各有千秋，但较为常用的是随机梯度下降。
        for i in range(m):
        	xi = dataMat[i]
        	yi = dataMat[i]
        	#判断是否是误分类样本
            #误分类样本特诊为： -yi(w*xi+b)>=0，详细可参考书中2.2.2小节
            #在书的公式中写的是>0，实际上如果=0，说明改点在超平面上，也是不正确的
            if -1 * yi * (w * xi.T + b) >= 0:
            	#对于误分类样本，进行梯度下降，更新w和b
            	w = w + h * yi * xi
            	b = b + h * yi
        print('Round %d:%d training' % (k, iter))
    return w, b

def model_test(dataArr, labelArr, w, b):

	print('start to test')

	dataMat = np.mat(dataArr)
	labelMat = np.mat(labelArr).T

	m,n  = np.shape(dataMat)
	#错误样本数计数
	errorCnt = 0

	for i in range(m):
		xi = dataMat[i]
		yi = labelMat[i]
		result = -1 * yi * (w * xi.T + b)
		if result >= 0: errorCnt += 1

	accruRate = 1 - (errorCnt / m)
    #返回正确率
    return accruRate

if __name__ = 'main':
	start = time.time()

	trainData, trainLabel = loadData('F:\\dataset\\mnist\\mnist_dataset\\mnist_dataset\\mnist_train.csv')

	testData,testLabel = loadData('F:\\dataset\\mnist\\mnist_dataset\\mnist_dataset\\mnist_test.csv')

	w, b = perception(trainData, trainLabel, iter = 30)
	accruRate = model_test(testData, testLabel, w, b)

	end = time.time()
	print('accuracy rate is:', accruRate)
	print('time span:', end - start)







