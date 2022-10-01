import random
import matplotlib.pyplot as plt
import seaborn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import  seaborn as se
data = pd.read_excel('taxi.xlsx')
pick = data.pickup
drop = data.dropoff
# for i in range (1,len(drop)):
#     pdate = str(pick[i]).replace(':',' ').replace('-',' ').split()
#     pdate = datetime.datetime(*map(int, pdate))
#     ddate = str(drop[i]).replace(':',' ').replace('-',' ').split()
#     ddate = datetime.datetime(*map(int, ddate))
#     delta = ddate - pdate
#     data['delivery'][i-1] = int(delta.total_seconds()/60)
# new_data = pd.unique(data)
# model = LinearRegression()
# model.fit(data[['total']],data['delivery'])
# pred = model.predict(data[['total']])
# print(model.coef_,model.intercept_)
# print(accuracy_score(data.total.astype(int),pred.astype(int))
# print(new_data)

from sklearn.svm import SVC
color = pd.get_dummies(data['color'],drop_first=True)
# print(color)
x = (data[['passengers','distance','total']])
x['colors'] = color
y = data['payment'].fillna('credit card')
y = pd.get_dummies(y,drop_first=True)
# x,xt,y, yt = train_test_split(x,y)
from acc_check import accuracy
# accuracy(x,xt,y, yt)
# model = SVC()
# model.fit(x,y)
# pre = model.predict(xt)
# print(accuracy_score(pre,yt))
# model = DecisionTreeClassifier()
# model.fit(x,y)
# pre = model.predict(xt)
# print(accuracy_score(pre,yt))
# model = RandomForestClassifier()
# model.fit(x,y)
# pre = model.predict(xt)
# print(accuracy_score(pre,yt))
# from sklearn.naive_bayes import *
# model = BernoulliNB()
# model.fit(x,y)
# pre = model.predict(xt)
# print(accuracy_score(pre,yt))
# model = GaussianNB()
# model.fit(x,y)
# pre = model.predict(xt)
# print(accuracy_score(pre,yt))
# model = MultinomialNB()
# model.fit(x,y)
# pre = model.predict(xt)
# print(accuracy_score(pre,yt))
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=1)
# model.fit(x,y)
# pre = model.predict(xt)
# print(accuracy_score(pre,yt))

fig, axs = plt.subplots(ncols=2, nrows=2)
from sklearn.cluster import KMeans
# import seaborn as se
# import matplotlib.pyplot as plt
new_data = data[['total','distance','passengers']]
cluster = KMeans(n_clusters=10)
# given = [0, 1, 2]
# desired = ['low','medium','high']
cluster.fit(new_data)
new_data['label'] =(cluster.labels_)
# low, high, med = new_data.groupby(by='label')
# print(low)
# print(med)
# print(high)
# print(len(low[1].total),len(med[1].total),len(high[1].total))
# print(data.shape)
# # sum = 0
# # for i in high[1].total:
# #     # print(i)
# #     sum += i
# # print(sum/len(high[1].total))
# # print(pd.DataFrame(high[1]).describe().mean().total)
# # print(pd.DataFrame(med[1]).describe().mean().total)
# # print(low[1].describe().mean().total)
se.scatterplot(data['distance'],data['total'],hue=new_data['label'],s=5,ax=axs[(0,1)])



# datas = se.get_dataset_names()
# data = se.load_dataset(datas[4])
# xd = data.drop(['abbrev'],axis=1)
# y=data['ins_premium']
# print(y)
# from sklearn.cluster import KMeans
# print(xd.columns)
# model = KMeans(n_clusters=3)
# model.fit(xd)
# xd['label'] = model.labels_
# se.scatterplot(x = 'ins_premium', y = 'ins_losses', data = xd ,hue= 'label')
# plt.tight_layout()
# plt.grid()
# plt.show()


from sklearn.cluster import  AgglomerativeClustering
agg = AgglomerativeClustering(n_clusters=10).fit(x).labels_
print(len(agg))
se.scatterplot(data['distance'],data['total'],hue=agg,s=5,ax=axs[(1,0)])
plt.style.use('ggplot')
plt.show()
from sklearn.cluster import  Birch
agg = Birch(threshold=1.7, n_clusters=10).fit(x).labels_
se.scatterplot(data['distance'],data['total'],hue=agg,s=5,ax=axs[(1,1)])
print('sucess')
# plt.style.use('ggplot')
plt.show()




