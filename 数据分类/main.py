import xlrd
import numpy as np
import random
import time

global result_all
result_all=0


wb = xlrd.open_workbook('E:\\sonar.xlsx')
sheet = wb.sheet_by_name("Sheet1")

data_all=np.zeros((208,61))
for i in range (208):
    data_all[i] = sheet.row_values(i)
data_train=np.zeros((84,61))    #取40%作为训练集


def DOING():
    global result_all
    count = 0
    p1 = []
    p2 = []
    buffer = random.sample(range(0,207),84)   #第一类数据中取得84个进行训练，取了总量的40%
    for i in range(84):                 #注：使用randint生成的数会有重复的，而使用sample才会生成不重复的随机数
        data_train[i] = data_all[buffer[i]]
    for i in range(84):
        if data_train[i][60]==1:
            count=count+1
    P1=count/84
    P2=1-P1
    a=(data_train.mean(axis=0))
    for i in range(60):
        count_buffer1=0
        count_buffer2=0
        for j in range(84):
            if data_train[j][60]==0:
                if data_train[j][i]>a[i]:
                    count_buffer1=count_buffer1+1
            if data_train[j][60]==1:
                if data_train[j][i]>a[i]:
                    count_buffer2=count_buffer2+1
        p1.append(count_buffer1/84)
        p2.append(count_buffer2/84)

    R=0 #判断正确的个数

    for i in range(208):
        pai1=1
        pai2=1
        for j in range(60):
            if data_all[i][j]>a[j]:
                pai1=pai1*p1[j]
                pai2=pai2*p2[j]
            else:
                pai1=pai1*(1-p1[j])
                pai2=pai2*(1-p2[j])
        if pai1>pai2:
            if data_all[i][60]==0:
                R=R+1
        else:
            if data_all[i][60]==1:
                R=R+1

    result=R/208
    result_all=result_all+result

if __name__ == '__main__':
    start = time.clock()
    for _ in range(10): #求算十次随机抽样取得结果的平均值
        DOING()
    elapsed = (time.clock() - start)
    print(result_all/10)
    print(elapsed/10)