from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB ,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

met = [DecisionTreeClassifier(),RandomForestClassifier(),LogisticRegression(),LinearRegression(),SVC(), GaussianNB(), BernoulliNB() ,MultinomialNB(), KNeighborsClassifier()]
accu = []
def accuracy(x,xt,y,yt):
    for i in met:
        try:
            model = i
            model.fit(x, y)
            pred = model.predict(xt)
            acc = accuracy_score(pred.astype(int), yt.astype(int))
        except:
            acc = 'error'
        accu.append(acc)
    # print(dict(zip(met, accu)))
    print(met[accu.index(max(accu))],' : ', max(accu))




