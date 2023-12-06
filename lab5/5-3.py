import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors

# найти ближайших пользователей по схожим интересам
# Загрузка данных о рейтингах фильмов от пользователей и данных о фильмах
df_rates = pd.read_csv(filepath_or_buffer='/home/valery/Рабочий стол/university/lab5/user_ratedmovies.dat', sep='\t')
df_movies = pd.read_csv(filepath_or_buffer='/home/valery/Рабочий стол/university/lab5/movies.dat', sep='\t', encoding='iso-8859-1')

# Создание LabelEncoder для пользователей и фильмов
enc_user = LabelEncoder()
enc_mov = LabelEncoder()

# Применение LabelEncoder к идентификаторам пользователей и фильмов
enc_user = enc_user.fit(df_rates.userID.values)
enc_mov = enc_mov.fit(df_rates.movieID.values)

# Отбор только тех фильмов, за которые голосовали пользователи
idx = df_movies.loc[:, 'id'].isin(df_rates.movieID)
df_movies = df_movies.loc[idx]

# Применение LabelEncoder для удобства работы с идентификаторами пользователей и фильмов
df_rates.loc[:, 'userID'] = enc_user.transform(df_rates.loc[:, 'userID'].values)
df_rates.loc[:, 'movieID'] = enc_mov.transform(df_rates.loc[:, 'movieID'].values)
df_movies.loc[:, 'id'] = enc_mov.transform(df_movies.loc[:, 'id'].values)

# Создание разреженной матрицы схожести рейтингов пользователей для фильмов
R = coo_matrix((df_rates.rating.values, (df_rates.userID.values, df_rates.movieID.values)))

# Применение метода SVD (Singular(разложение) Value Decomposition,) для разложения матрицы R
# простыми словами большую матрицу раскладывает на три более простые
u, s, vt = svds(R, k=6)

# Инициализация метода ближайших соседей
nn = NearestNeighbors(n_neighbors=10)
u = u[:, 1:]  # Exclude the bias term for users
nn.fit(u)

# Нахождение 10 ближайших соседей для пользователя №1 на основе матрицы u
_, ind = nn.kneighbors(u[1].reshape(1, -1))

# Создание таблицы с ближайшими по схожести пользователями
user_ids = df_rates.sort_values('userID').loc[:, 'userID'].unique()
cols = ['user'] + ['nn_{}'.format(i) for i in range(1, 10)]
df_ind_nn = pd.DataFrame(data=user_ids[ind], columns=cols)

# Вывод ближайших пользователей к пользователю №1
print('Ближайшие пользователи к пользователю №1')
print(df_ind_nn.head())
