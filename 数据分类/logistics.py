import xlrd
import numpy as np
import random
import time

global R
R=0

wb = xlrd.open_workbook('E:\\sonar.xlsx')
sheet = wb.sheet_by_name("Sheet1")

#采用交叉验证，取40%出来作为训练集，最后在全部数据上面进行准确率测试
data_all=np.zeros((208,61))
for i in range (208):
    data_all[i] = sheet.row_values(i)
data_train=np.zeros((84,61))
label=data_all[:,60]
data=np.delete(data_all,60,axis=1)
buffer = random.sample(range(0,207),84)   #第一类数据中取得84个进行训练，取了总量的40%
for i in range(84):                 #注：使用randint生成的数会有重复的，而使用sample才会生成不重复的随机数
    data_train[i] = data_all[buffer[i]]
label_train=data_train[:,60]
data_train=np.delete(data_train,60,axis=1)


def sigmoid(z):
    return 1/(1+np.exp(-z))

def gradAscent(data, labels):

    dataMatrix = data
    labelMat =np.mat(labels).T
    m,n = np.shape(dataMatrix)
    alpha = 0.001       # 设置步长
    times = 500     # 设置循环次数
    weights = np.ones((n,1))    #初始化，因为要进行乘法运算故全部为1

    for k in range(times):
        h = sigmoid(np.dot(dataMatrix,weights))
        error = (labelMat - h)  #误差处理
        weights = weights + np.dot(alpha * dataMatrix.T , error)
    return weights

def classfiy(array,weights,label):
    global R
    s=sigmoid(sum(np.dot(array,weights)))
    if s<0.5:
        if label==0:
            R=R+1
    else:
        if label==1:
            R=R+1


if __name__ == '__main__':
    start=time.clock()
    weights = gradAscent(data_train,label_train) 
    for i in range(208):
        classfiy(data[i],weights,label[i])
    elapsed = (time.clock() - start)
    print(R/208)
    print(elapsed)
