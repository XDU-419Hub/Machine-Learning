import xlrd
import numpy as np
import random
from sklearn.metrics import accuracy_score
import time

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

global mu_positive # 正样本的高斯分布的均值向量
global mu_negetive
global sigma
mu_positive = 0  # 正样本的高斯分布的均值向量
mu_negetive = 0
sigma=[]        #协方差矩阵

def dividedata(Train_Data,Train_Label):
    postive_num = 0  # 正样本个数
    negetive_num = 0  # 负样本个数
    global sigma
    global mu_positive  # 正样本的高斯分布的均值向量
    global mu_negetive

    postive_data = []
    negetive_data = []
    for (data,label) in zip(Train_Data,Train_Label):
        if label == 1:
            postive_num += 1
            postive_data.append(list(data))
        else:
            negetive_num += 1
            negetive_data.append(list(data))

    row,col = Train_Data.shape
    postive = postive_num*1.0/row
    negetive = 1-postive

    postive_data = np.array(postive_data)
    negetive_data = np.array(negetive_data)
    postive_data_sum = np.sum(postive_data, 0)
    negetive_data_sum = np.sum(negetive_data, 0)
    mu_positive = postive_data_sum/postive_num
    mu_negetive = negetive_data_sum/negetive_num              # 负样本的高斯分布的均值向量

    positive_deta = postive_data-mu_positive
    negetive_deta = negetive_data-mu_negetive

    for deta in positive_deta:
        deta = deta.reshape(1,col)
        ans = deta.T.dot(deta)
        sigma.append(ans)
    for deta in negetive_deta:
        deta = deta.reshape(1,col)
        ans = deta.T.dot(deta)
        sigma.append(ans)
    sigma = np.sum(np.array(sigma),0)
    sigma = sigma/row
    mu_positive = mu_positive.reshape(1,col)
    mu_negetive = mu_negetive.reshape(1,col)

def Gaussian(x, mean, cov):

    dim = np.shape(cov)[1]
    # cov的行列式为零时的措施
    covdet = np.linalg.det(cov + np.eye(dim) * 0.001)
    covinv = np.linalg.inv(cov + np.eye(dim) * 0.001)
    xdiff = (x - mean).reshape((1, dim))
    prob = 1.0 / (np.power(np.power(2 * np.pi, dim) * np.abs(covdet), 0.5)) * \
            np.exp(-0.5 * xdiff.dot(covinv).dot(xdiff.T))[0][0]
    return prob

def predict(test_data):
    global mu_positive
    global mu_negetive
    global sigma
    dividedata(data_train,label_train)
    predict_label = []
    for i in test_data:
        positive_pro = Gaussian(i,mu_positive,sigma)
        negetive_pro = Gaussian(i,mu_negetive,sigma)
        if positive_pro >= negetive_pro:
            predict_label.append(1)
        else:
            predict_label.append(0)
    return predict_label

if __name__ == '__main__':
    start=time.clock()
    test_predict = predict(data)
    elapsed = (time.clock() - start)
    print("GDA的正确率为：", accuracy_score(label, test_predict))
    print(elapsed)