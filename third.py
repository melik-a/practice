#DBSCAN кластеризация
import matplotlib.pyplot as plt


from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
#Генерация двумерного набора данных (в виде двух лун)
X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=0)
#Кластеризация DBSCAN
db = DBSCAN(eps=0.2,
            min_samples=5)
y_db = db.fit_predict(X)
#Отрисовка
plt.scatter(X[y_db==0,0],
            X[y_db==0,1],
            c='lightblue',
            marker='o',
            s=40,
            edgecolors='black',
            label='кластер 1')
plt.scatter(X[y_db==1,0],
            X[y_db==1,1],
            c='red',
            marker='s',
            s=40,
            edgecolors='black',
            label='кластер 2')
plt.legend()
plt.show()