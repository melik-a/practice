#k-means clustering
import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#Создаём двумерный набор данных из 150 точек
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  random_state=0)
#Создание графика
fig = plt.figure(1)
plt.scatter(X[:,0],
            X[:,1],
           c='white',edgecolors='black',
           marker='o',
           s=50)
plt.grid()

#Кластеризация методом к-средних (первоначальные центроиды выбираются методом k-means++)
kmeans = KMeans(n_clusters=3,init='k-means++',n_init=10,max_iter=300,tol=1e-04,random_state=0)
y_kmeans = kmeans.fit_predict(X)

fig2 = plt.figure(2)
plt.scatter(X[y_kmeans==0,0],
            X[y_kmeans==0,1],
            s=50,
            c='green',
            edgecolors='black',
            marker='s')
plt.scatter(X[y_kmeans==1,0],
            X[y_kmeans==1,1],
            s=50,
            c='orange',
            edgecolors='black',
            marker='o')
plt.scatter(X[y_kmeans==2,0],
            X[y_kmeans==2,1],
            s=50,
            c='blue',
            edgecolors='black',
            marker='v')
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            s=250,
            marker='*',
            c='red',
            edgecolors='black')

plt.legend(["кластер 1", "кластер 2", "кластер 3","центроиды"])
plt.grid()
#plt.show()
print('Искажение : % .2f' % kmeans.inertia_)

#Используем метод локтя для нахождения оптимального кол-ва кластеров (на основе искажения)
fig3 = plt.figure()
distortion = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    distortion.append(kmeans.inertia_)
plt.plot(range(1,11),distortion,marker='o')
plt.xlabel('Число кластеров')
plt.ylabel('Искажение')
plt.show()
