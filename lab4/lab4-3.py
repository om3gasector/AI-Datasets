# поиск конечных результатов по признакам
# Импортируем необходимые библиотеки
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

# Загружаем данные о жилье в Калифорнии
data = fetch_california_housing()
X = data.data
y = data.target

# Масштабируем признаки
X_scaled = scale(X)

# Определяем значения параметра p для оценки
p_values = np.linspace(1, 20, 50)

# Инициализируем переменные для лучшего значения p и среднеквадратичного отклонения
best_p = None
best_mse = float('inf')

# Перебираем значения р - параметра, который определяет, как измеряется расстояние между соседними точками 
for p in p_values:
    # Создаем модель KNeighborsRegressor с текущим значением p
    knn = KNeighborsRegressor(n_neighbors=6, weights='distance', p=p)
    
    # Вычисляем среднеквадратичное отклонение с использованием кросс-валидации
    mse_scores = -cross_val_score(knn, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
    
    # Вычисляем среднее среднеквадратичное отклонение
    mean_mse = mse_scores.mean()
    
    # Печатаем текущее значение p
    print(p)
    
    # Обновляем лучшие параметры, если текущее среднеквадратичное отклонение лучше
    if mean_mse < best_mse:
        best_mse = mean_mse
        best_p = p

# Выводим лучшее значение p и среднеквадратичное отклонение
print("\nЛучшее значение p:", best_p)
print("\nЛучшее среднеквадратичное отклонение:", best_mse)
