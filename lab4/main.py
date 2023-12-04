# Импортируем необходимые библиотеки
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Загружаем данные из файла "titanic.csv" в объект DataFrame
data = pd.read_csv("/home/valery/Рабочий стол/university/MachineDeepLearning/introductory_laboratory_work (№1)/Dataset/titanic.csv")

# Выбираем определенные признаки для анализа
selected_features = ['Pclass', 'Fare', 'Age', 'Sex', 'Survived']
data = data[selected_features]

# Преобразуем категориальный признак "Sex" в числовой формат
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)

# Удаляем строки с отсутствующими значениями
data = data.dropna()

# DecisionTreeClassifier
# Создаем матрицу признаков X и вектор целевой переменной y
X = data.drop('Survived', axis=1)
y = data['Survived']

# Создаем экземпляр класса DecisionTreeClassifier с параметром random_state=100
clf = DecisionTreeClassifier(random_state=100)

# Обучаем модель на данных
clf.fit(X, y)

# Вычисляем важность признаков
feature_importance = clf.feature_importances_

# Получаем имена признаков
feature_names = X.columns

# Создаем словарь, связывающий имена признаков с их важностью
importance_dict = dict(zip(feature_names, feature_importance))

# Сортируем словарь в убывающем порядке важности признаков
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

# Выводим два наиболее важных признака
top_features = sorted_importance[:2]
print("Самые важные признаки:\n")
for feature, importance in top_features:
    print(f"{feature}: {importance}")
