import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # 载入 pyplot API


 
def compare(a,b):
    if a<b:
        print("ANN is more accurate whose accuracy is :",b)
    elif a>b:
        print("DT is more accurate whose accuracy is:",a)
    else:
        print("ANN and DT has the same accuracy :",a)
    return 0
'''
def getNumLeafs(myTree):
    #初始化树的叶子节点个数
    numLeafs = 0
    #myTree.keys()获取树的非叶子节点'no surfacing'和'flippers'
    #list(myTree.keys())[0]获取第一个键名'no surfacing'
    firstStr = list(myTree.keys())[0]
    #通过键名获取与之对应的值，即{0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}
    secondDict = myTree[firstStr]
    #遍历树，secondDict.keys()获取所有的键
    for key in secondDict.keys():
        #判断键是否为字典，键名1和其值就组成了一个字典，如果是字典则通过递归继续遍历，寻找叶子节点
        if type(secondDict[key]).__name__=='dict':
            numLeafs += getNumLeafs(secondDict[key])
        #如果不是字典，则叶子结点的数目就加1
        else:
            numLeafs += 1
    #返回叶子节点的数目
    return numLeafs

def getTreeDepth(myTree):
    #初始化树的深度
    maxDepth = 0
    #获取树的第一个键名
    firstStr = list(myTree.keys())[0]
    #获取键名所对应的值
    secondDict = myTree[firstStr]
    #遍历树
    for key in secondDict.keys():
        #如果获取的键是字典，树的深度加1
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        #去深度的最大值
        if thisDepth > maxDepth : maxDepth = thisDepth
    #返回树的深度
    return maxDepth

#设置画节点用的盒子的样式
decisionNode = dict(boxstyle = "sawtooth",fc="0.8")
leafNode = dict(boxstyle = "round4",fc="0.8")
#设置画箭头的样式    http://matplotlib.org/api/patches_api.html#matplotlib.patches.FancyArrowPatch
arrow_args = dict(arrowstyle="<-")
#绘图相关参数的设置
def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    

    createPlot.ax1.annotate(nodeTxt,xy=parentPt,\
    xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',\
    va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
 
#绘制线中间的文字(0和1)的绘制
def plotMidText(cntrPt,parentPt,txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]   #计算文字的x坐标
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]   #计算文字的y坐标
    createPlot.ax1.text(xMid,yMid,txtString)
#绘制树
def plotTree(myTree,parentPt,nodeTxt):
    #获取树的叶子节点
    numLeafs = getNumLeafs(myTree)
    #获取树的深度
    depth = getTreeDepth(myTree)
    #firstStr = myTree.keys()[0]
    #获取第一个键名
    firstStr = list(myTree.keys())[0]
    #计算子节点的坐标
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,\
              plotTree.yoff)
    #绘制线上的文字
    plotMidText(cntrPt,parentPt,nodeTxt)
    #绘制节点
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    #获取第一个键值
    secondDict = myTree[firstStr]
    #计算节点y方向上的偏移量，根据树的深度
    plotTree.yoff = plotTree.yoff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            #递归绘制树
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            #更新x的偏移量,每个叶子结点x轴方向上的距离为 1/plotTree.totalW
            plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
            #绘制非叶子节点
            plotNode(secondDict[key],(plotTree.xoff,plotTree.yoff),\
                     cntrPt,leafNode)
            #绘制箭头上的标志
            plotMidText((plotTree.xoff,plotTree.yoff),cntrPt,str(key))
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD
 
#绘制决策树，inTree的格式为{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
def createPlot(inTree):
    #新建一个figure设置背景颜色为白色
    fig = plt.figure(1,facecolor='white')
    #清除figure
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    #创建一个1行1列1个figure，并把网格里面的第一个figure的Axes实例返回给ax1作为函数createPlot()
    #的属性，这个属性ax1相当于一个全局变量，可以给plotNode函数使用
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    #获取树的叶子节点
    plotTree.totalW = float(getNumLeafs(inTree))
    #获取树的深度
    plotTree.totalD = float(getTreeDepth(inTree))
    #节点的x轴的偏移量为-1/plotTree.totlaW/2,1为x轴的长度，除以2保证每一个节点的x轴之间的距离为1/plotTree.totlaW*2
    plotTree.xoff = -0.5/plotTree.totalW
    plotTree.yoff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
    
createPlot(y_text)
'''




'''
===============================================================================
'''
start = datetime.datetime.now()
col_names = ['winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
pima = pd.read_csv("new_data.csv",header=None, names=col_names) 
pima = pima.iloc[1:] # delete the first row of the dataframe

feature_cols = ['firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
x = pima[feature_cols] # Features
y = pima.winner # Target variable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=1) 
# 80% training and 20% test

clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)# Create Decision Tree classifer object
clf = clf.fit(x_train,y_train)# Train Decision Tree Classifer
y_pred = clf.predict(x_test)#Predict the response for test dataset
#print(y_pred)
a = accuracy_score(y_test, y_pred)

end = datetime.datetime.now()
t = end-start

start = datetime.datetime.now()
pima2 = pd.read_csv("test_set.csv")
m = pima2.drop(['gameId','creationTime','gameDuration','winner'],axis=1).values
n = pima2['winner'].values
m_train, m_test, n_train, n_test = train_test_split(m, n, test_size=0.2,random_state=42)

'''
#m_train = torch.tensor(m_train.values, dtype=int())
m_train = torch.FloatTensor(m_train.values.astype(np.int8))
m_test = torch.FloatTensor(m_test.values.astype(np.int8))
n_train = torch.LongTensor(n_train.values.astype(np.int8))
n_test = torch.LongTensor(n_test.values.astype(np.int8))
'''

m_train = torch.FloatTensor(m_train)
m_test = torch.FloatTensor(m_test)
n_train = torch.LongTensor(n_train)
n_test = torch.LongTensor(n_test)

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=17, out_features=30)
        self.output = nn.Linear(in_features=30, out_features=3)
    def forward(self, m):
        m = torch.sigmoid(self.fc1(m))
        m = self.output(m)
        m = F.softmax(m)
        return m

model = ANN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 100
loss_arr = []
for i in range(epochs):
    n_hat = model.forward(m_train)
    loss = criterion(n_hat, n_train)
    loss_arr.append(loss)
    if i%2 == 0:
        print(f'Epoch:{i} Loss:{loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

predict_out = model(m_test)
_,predict_n = torch.max(predict_out, 1)
b =  accuracy_score(n_test, predict_n)


end = datetime.datetime.now()
print(t)
print (end -start)
print("a:",a,"b:",b/n)
compare(a,b)









