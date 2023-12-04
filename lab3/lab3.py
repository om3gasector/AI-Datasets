# импорт библиотек и модулей для работы с данными, кластеризации, метриками и визуализации
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

# загрузка данных с использованием load_digits
data = load_digits()
data_scaled = scale(data.data)

# вывод базовых характеристик данных, таких как:
print("\nРазмерность данных:", data.data.shape)
print("\nКоличество признаков:", data.data.shape[1])
print("\nКоличество объектов:", data.data.shape[0])
print("\nКоличество уникальных значений в load_digits().target:", len(np.unique(data.target)))

# выполняется кластеризация данных с использованием алгоритма K-Means с инициализацией k-means++
# k-means - алгоритм кластеризации, который разделяет набор данных на K кластеров на основе схожести объектов
kmeans_pp = KMeans(init='k-means++', n_clusters=len(np.unique(data.target)), n_init=10).fit(data_scaled)

# вычисляются метрики качества кластеризации, такие как ARI (Adjest Rand Index) AMI (Adjusted Mutual Information) для K-Means

# ARI измеряет сходство между результатами кластеризации и истинными метками, учитывая случайные совпадения
# AMI почти тоже самое, но он учитывает общую информацию между кластерами и истинными классами.
ari_pp = adjusted_rand_score(data.target, kmeans_pp.labels_)
ami_pp = adjusted_mutual_info_score(data.target, kmeans_pp.labels_)

print("\nARI для KMeans с init='k-means++':", ari_pp)
print("\nAMI для KMeans с init='k-means++':", ami_pp)

print("\nВремя работы алгоритма для KMeans с init='k-means++':", kmeans_pp.n_iter_)

# аналогично  23 строке, но используются K-Means с инициализацией random, +вычисляются метрики качества кластеризации
kmeans_random = KMeans(init='random', n_clusters=len(np.unique(data.target)), n_init=10).fit(data_scaled)
ari_random = adjusted_rand_score(data.target, kmeans_random.labels_)
ami_random = adjusted_mutual_info_score(data.target, kmeans_random.labels_)

print("\nARI для KMeans с init='random':", ari_random)
print("\nAMI для KMeans с init='random':", ami_random)

print("\nВремя работы алгоритма для KMeans с init='random':", kmeans_random.n_iter_)

# была ошибка: проделано снижение размерности данных с использованием метода PCA с 64 компонентами
pca = PCA(n_components=64)
data_pca = pca.fit_transform(data_scaled)

# Выбираем первые три компонента PCA для трехмерной визуализации
data_3d = data_pca[:, :3]

# выполнение кластеризации данных с использованием K-Means с инициализацией, основанной на компонентах PCA + вычисление метрик
kmeans_pca = KMeans(init='k-means++', n_clusters=len(np.unique(data.target)), n_init=10).fit(data_scaled)
ari_pca = adjusted_rand_score(data.target, kmeans_pca.labels_)
ami_pca = adjusted_mutual_info_score(data.target, kmeans_pca.labels_)

print("\nARI для KMeans с init='k-means++':", ari_pca)
print("\nAMI для KMeans с init='k-means++':", ami_pca)

print("\nВремя работы алгоритма для KMeans с init='k-means++':", kmeans_pca.n_iter_, "\n")


# визуализация для KMeans с init='k-means++'
fig = plt.figure(figsize=(12, 12))  # создаем только один объект figure

# осуществляем кластеризации в трех измерениях
kmeans_pp_3d = KMeans(init='k-means++', n_clusters=len(np.unique(data.target)), n_init=10).fit(data_3d)
cluster_centers_pp_3d = kmeans_pp_3d.cluster_centers_

ax1 = fig.add_subplot(111, projection='3d') 
ax1.set_title('KMeans++, PCA (3D)')

scatter = ax1.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=kmeans_pp_3d.labels_, edgecolor='none', alpha=0.7, cmap=plt.cm.get_cmap('nipy_spectral', 10))
ax1.scatter(cluster_centers_pp_3d[:, 0], cluster_centers_pp_3d[:, 1], cluster_centers_pp_3d[:, 2], marker='x', s=100, c='black')
fig.colorbar(scatter)

plt.show()