import pandas as pd

pd.options.mode.chained_assignment = None # default="warn"

#указание пути к датасету
data = pd.read_csv("/home/valery/Рабочий стол/university/MachineDeepLearning/introductory_laboratory_work (№1)/Dataset/titanic.csv")
#количество мужчин
male_passengers = data[data["Sex"] == "male"]
male_count = len(male_passengers)
print(f"\nКолличество мужчин: {male_count}")
#количество выживших в процентах
survival_rate = (data["Survived"].sum() / len(data)) * 100
print(f"\nКоличество выживших: {survival_rate:.2f}")
#доля пассажиров во 2 классе в процентах
second_class_percentagers_rate = (len(data[data['Pclass'] == 2]) / len(data)) * 100
print(f"\nВо 2-м классе доля пассажиров составляет: {second_class_percentagers_rate:.2f}")
#средний и медианный возраст
average_age = data["Age"].mean()
median_age = data["Age"].median()
print(f"\nСредний возраст составляет: {average_age:.2f} лет")
print(f"\nМедианный возраст составляет: {median_age} лет")

#отображения взаимосвязи / соотношения между:
#количеством (братьев и сестёр / супругов на борту) и (Количество родителей / детей на борту)  
pearson_correlation_sibsp_parch = data["SibSp"].corr(data["Parch"])
print(f"\nKoрреляция SibSp и Parch: {pearson_correlation_sibsp_parch:.2f}")
#функция для извлечения самого популярного женского имени
def extract_first_name(name):
        titles = ["Miss", "Dr", "Master", "Don"]
        for title in titles:
            name = name.replace(title, "")
        name = name. split(",")[1].split(". ")[1].strip()
        return name

female_passengers = data[data["Sex"] == "female"]
female_passengers["firstName"] = female_passengers["Name"].apply(extract_first_name)
most_common_name = female_passengers["firstName"].mode().values[0]
print(f"\nПопулярным женским именем является: {most_common_name}\n")