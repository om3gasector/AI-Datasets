# метод ближайших соседей, найти оптимальное значение k
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

# Загружаем данные из файла 'titanic.csv' в объект DataFrame
data = pd.read_csv("/home/valery/Рабочий стол/university/MachineDeepLearning/introductory_laboratory_work (№1)/Dataset/titanic.csv")

# Удаляем строки с отсутствующими значениями
data = data.dropna(subset=['Age', 'SibSp', 'Parch', 'Fare'])

# Определяем матрицу признаков X и вектор целевой переменной y
X = data.select_dtypes(include=['number']).drop(columns=['Survived'])  # Используем только числовые признаки
y = data['Survived']

# Создаем объект KFold для кросс-валидации
kf = KFold(n_splits=10, shuffle=True, random_state=100)

# Оцениваем качество модели без масштабирования
k_values = range(1, 101)
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    try:
        score = cross_val_score(knn, X, y, cv=kf, scoring='accuracy').mean()
        scores.append(score)
    except ValueError:
        print(f"Skipping k={k} due to insufficient data for cross-validation.")

if scores:
    best_k = k_values[scores.index(max(scores))]
    best_accuracy = max(scores)
    print(f"\nЛучшее значение k без масштабирования: {best_k}, \nКачество: {best_accuracy}")
else:
    print("Нет данных для расчета качества без масштабирования.")
