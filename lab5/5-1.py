from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np

# Apriori & FP-growth - алгоритмы для обнаружения частых набором элеметов в наборе данных
# Загрузка данных (предполагается, данные в формате списка транзакций)
dataset = [
    ['apple', 'beer', 'rice', 'chicken'], ['apple', 'beer', 'rice'],
    ['apple', 'beer'], ['apple', 'banana'], ['bread', 'butter', 'milk'],
    ['bread', 'butter'], ['milk', 'sugar'], ['coffee', 'sugar'],
    ['bread', 'butter', 'coffee'], ['bread', 'milk', 'sugar'],
    ['apple', 'beer', 'bread', 'butter', 'rice', 'chicken'],
    ['apple', 'beer', 'bread', 'butter', 'rice'],
    ['coffee', 'milk'], ['bread', 'butter', 'sugar'],
    ['apple', 'bread', 'milk']
] 

# Преобразование данных в формат, понятный библиотеке mlxtend
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Оценка времени выполнения Apriori
apriori_times = []
for support in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    start_time = time.time()
    frequent_itemsets_apriori = apriori(df, min_support=support, use_colnames=True)
    end_time = time.time()
    apriori_times.append(end_time - start_time)

# Оценка времени выполнения FP-growth
fpgrowth_times = []
for support in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    start_time = time.time()
    frequent_itemsets_fpgrowth = fpgrowth(df, min_support=support, use_colnames=True)
    end_time = time.time()
    fpgrowth_times.append(end_time - start_time)

# Вывод результатов
results_df = pd.DataFrame({
    'Support': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'Apriori Time': apriori_times,
    'FP-growth Time': fpgrowth_times
})

# Вывод результатов в таблицу
print(results_df)

# Построение графиков
plt.plot(results_df['Support'], results_df['Apriori Time'], label='Apriori')
plt.plot(results_df['Support'], results_df['FP-growth Time'], label='FP-growth')
plt.xlabel('Support (%)')
plt.ylabel('Ln(Time)')
plt.legend()
plt.show()
