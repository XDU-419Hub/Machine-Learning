from scipy.io import loadmat
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest,chi2,SelectFromModel,RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import time
import warnings
from sklearn.metrics import accuracy_score,f1_score,precision_score



warnings.filterwarnings("ignore")
m=loadmat('Indian_pines_corrected.mat')
n=loadmat('Indian_pines_gt.mat')
data=np.array(m['indian_pines_corrected'])
label=np.array(n['indian_pines_gt'])

data=data.flatten()
data=np.reshape(data,(21025,200))
label=label.flatten()


def hf(x,y):                                #划分训练集和测试集，返回训练集和测试集
    x_1,x_2,y_1,y_2=train_test_split(x,y,test_size=0.6)
    return x_1,x_2,y_1,y_2

def classfiy(train_x,train_y,test_x,test_y):      #采用KNN分类器，输出分类评价
    clf = KNeighborsClassifier()
    clf.fit(train_x, train_y)
    y_p=clf.predict(test_x)
    print('准确率为：',accuracy_score(test_y, y_p))
    #'weighted' 为每个标签计算指标，并通过各类占比找到它们的加权均值（每个标签的正例数）.它解决了’macro’的标签不平衡问题；它可以产生不在精确率和召回率之间的F-score.
    print('精确率为：',precision_score(test_y,y_p,average='weighted'))
    print('f1 score为：',f1_score(test_y,y_p,average='weighted'))  #除计算方式不同，参数含义相同

def selection1(data,label):      #卡方检验,Filter
    data_new=SelectKBest(chi2,k=40).fit_transform(data,label)   #保留40维数据
    return data_new

def selection2(data,label):     #递归特征消除，Wrapper
    data_new=[]
    lr=LogisticRegression()
    rfe=RFE(estimator=lr,n_features_to_select=40,step=20)       #最后保留40维数据，每次迭代去掉20维
    rfe.fit(data,label)
    for i in range(200):
        if rfe.support_[i]==True:
            data_new.append(data[:,i])
    data_new=np.array(data_new)
    data_new=data_new.T
    return data_new

def selection3(data,label):     #基于树的特征选择,Embedded
    clf=ExtraTreesClassifier()
    clf=clf.fit(data,label)
    model=SelectFromModel(clf,threshold=0.0062,prefit=True)      #threshold是阈值，默认是均值即0.005，0.0062保证数据在40左右
    data_new=model.transform(data)
    return data_new


if __name__ == '__main__':
    train_data,test_data,train_label,test_label=hf(data,label)
    classfiy(train_data,train_label,test_data,test_label)
    print(' ')

    start=time.clock()
    data1=selection1(data,label)
    end=time.clock()
    print('filter型用时：',(end-start))
    train_data, test_data, train_label, test_label = hf(data1, label)
    classfiy(train_data,train_label,test_data,test_label)
    print(' ')

    start=time.clock()
    data2=selection2(data, label)
    end=time.clock()
    print('Wrapper型用时：',(end-start))
    train_data, test_data, train_label, test_label = hf(data2, label)
    classfiy(train_data,train_label,test_data,test_label)
    print(' ')

    start=time.clock()
    data3=selection3(data,label)
    end=time.clock()
    print('Embedded型用时：',(end-start))
    train_data, test_data, train_label, test_label = hf(data3, label)
    classfiy(train_data,train_label,test_data,test_label)
