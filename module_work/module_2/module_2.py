import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import ttest_ind

import warnings

warnings.filterwarnings("ignore")


#  универсаклная функция замены пропуска  none
def replaceEmptySkipValue(data, columns):
    for column in columns:
        data[column] = data[column].apply(
            lambda x: None if pd.isnull(x) else (None if (x == 'nan' or str(x).strip() == '') else x))
    return data


# преобразование незвестного значения
def ordanaryValidValue(data, column, needOutliersData=False):
    data = replaceEmptySkipValue(data, [column])

    data = determNoneValue(data, column)

    if needOutliersData:
        data = outliersData(data, column)

    return data


# фильтрация данных на валидных значениях + значение по умочанию
def permitValidValue(data, column, premit_value, defaulVaue=None, needOutliersData=False):
    data = replaceEmptySkipValue(data, [column])

    data[column] = data[column].apply(lambda x: x if x in premit_value else defaulVaue)

    data = determNoneValue(data, column)
    if needOutliersData:
        data = outliersData(data, column)
    return data


# фильтрация положительных данных
def permitPositivValue(data, column, defaulVaue=None, needOutliersData=False):
    data = replaceEmptySkipValue(data, [column])
    data[column] = data[column].apply(lambda x: int(x) if x >= 0 else defaulVaue)

    data = determNoneValue(data, column)
    if needOutliersData:
        data = outliersData(data, column)

    return data


# замена пропусков данных модами
def determNoneValue(data, column):
    value_ = data[column].mode()[0]
    data.loc[:, column].fillna(value_, inplace=True)
    return data


#  функция фильтраци выбросов - если квинтили равны  -  не делаем выросы - распределени примено одинаково
def outliersData(data, column):
    quantile_3 = data[column].quantile(0.75)
    quantile_1 = data[column].quantile(0.25)
    if quantile_3 == quantile_1:
        return data

    IQR = quantile_3 - quantile_1
    column_min = quantile_1 - 1.5 * IQR
    column_max = quantile_3 + 1.5 * IQR

    data = data[data[column].between(column_min, column_max)]
    return data


student = pd.read_csv("C:/study/skillFactory/module_2/stud_math.xls")
#  неизвестный столбец - лишнеи данные не помешают
student.rename(columns={'studytime, granular': 'granular'}, inplace=True)
student['granular'] = student['granular'] * (-1)

# TODO Числовые данные   granular freetime goout health absences score
# age - выбросов нет
student = permitPositivValue(student, 'age')
# student['age'].hist()

# granular - неизвестный параметр
student = ordanaryValidValue(student, 'granular', needOutliersData=True)
# student['age'].hist()

# Medu - образование матери , доступыне значения (0,1,2,3,4)
student = permitValidValue(student, 'Medu', [0, 1, 2, 3, 4], needOutliersData=True)
# student['Medu'].hist()

# Fedu - образование отца , доступыне значения (0,1,2,3,4)
student = permitValidValue(student, 'Fedu', [0, 1, 2, 3, 4], needOutliersData=True)
# student['Fedu'].hist()

# traveltime - время в пути до школы
student = permitValidValue(student, 'traveltime', [1, 2, 3, 4], needOutliersData=True)
# student['traveltime'].hist()

# studytime  - время на учёбу помимо школы в неделю
student = permitValidValue(student, 'studytime', [1, 2, 3, 4], needOutliersData=True)
# student['studytime'].hist()

# failures   - количество внеучебных неудач
student = permitValidValue(student, 'failures', [1, 2, 3], defaulVaue=0, needOutliersData=True)
# student['failures'].hist()

# famrel - семейные отношения
student = permitValidValue(student, 'famrel', [1, 2, 3, 4, 5], needOutliersData=True)
# student['famrel'].hist()

# freetime   - свободное время после школы
student = permitValidValue(student, 'freetime', [1, 2, 3, 4, 5], needOutliersData=True)
# student['freetime'].hist()

# goout - проведение времени с друзьями
student = permitValidValue(student, 'goout', [1, 2, 3, 4, 5], needOutliersData=True)
# student['goout'].hist()

# goout - текущее состояние здоровья
student = permitValidValue(student, 'health', [1, 2, 3, 4, 5], needOutliersData=True)
# student['health'].hist()

# absences  -  количество пропущенных занятий
student = permitPositivValue(student, 'absences', needOutliersData=True)
# student['absences'].hist()

# score — баллы по госэкзамену по математике
student = permitPositivValue(student, 'score', needOutliersData=True)
# student['score'].hist()


# TODO  уникальных значений для номинативных переменных
# sex — пол ученика ('F' - женский, 'M' - мужской)
student = permitValidValue(student, 'sex', ['F', 'M'])
# student['sex'].hist()

# address — тип адреса ученика ('U' - городской, 'R' - за городом)
student = permitValidValue(student, 'address', ['U', 'R'])
# student['address'].hist()

# famsize — размер семьи('LE3' <= 3, 'GT3' >3)
student = permitValidValue(student, 'famsize', ['LE3', 'GT3'])
# student['famsize'].hist()

# Pstatus - статус совместного жилья родителей ('T' - живут вместе 'A' - раздельно)
student = permitValidValue(student, 'Pstatus', ['T', 'A'])
# student['Pstatus'].hist()

# Mjob - работа матери
student = permitValidValue(student, 'Mjob', ['teacher', 'health', 'services', 'at_home', 'other'])
# student['Mjob'].hist()

# Fjob  - работа отца
student = permitValidValue(student, 'Fjob', ['teacher', 'health', 'services', 'at_home', 'other'])
# student['Fjob'].hist()

# reason  — причина выбора школы
student = permitValidValue(student, 'reason', ['home', 'reputation', 'course', 'other'])
# student['reason'].hist()

# guardian  — опекун
student = permitValidValue(student, 'guardian', ['mother', 'father', 'other'])
# student['guardian'].hist()

# schoolsup - дополнительная образовательная поддержка
student = permitValidValue(student, 'schoolsup', ['yes', 'no'])
# student['guardian'].hist()

# famsup — семейная образовательная поддержка
student = permitValidValue(student, 'famsup', ['yes', 'no'])
# student['famsup'].hist()

# paid — дополнительные платные занятия по математике
student = permitValidValue(student, 'paid', ['yes', 'no'])
# student['paid'].hist()

# activities — дополнительные внеучебные занятия
student = permitValidValue(student, 'activities', ['yes', 'no'])
# student['activities'].hist()

# nursery — посещал детский сад
student = permitValidValue(student, 'nursery', ['yes', 'no'])
# student['nursery'].hist()

# higher — хочет получить высшее образование
student = permitValidValue(student, 'higher', ['yes', 'no'])
# student['higher'].hist()

# internet — наличие интернета дома
student = permitValidValue(student, 'internet', ['yes', 'no'])
# student['internet'].hist()

# romantic — в романтических отношениях
student = permitValidValue(student, 'romantic', ['yes', 'no'])
# student['romantic'].hist()

correlation = student.corr()
# sns.pairplot(student, kind = 'reg')

print(student['studytime'].value_counts())
exit()

def get_boxplot(date, column):
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.boxplot(x=column, y='score',
                data=date.loc[:],
                ax=ax)
    plt.xticks(rotation=45)
    ax.set_title('Boxplot for ' + column)
    plt.show()


for col in ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health',
            'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
            'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
            'romantic', 'absences']:
    get_boxplot(student, col)


def get_stat_dif(date, column):
    cols = date.loc[:, column].value_counts().index[:10]
    combinations_all = list(combinations(cols, 2))
    for comb in combinations_all:
        if ttest_ind(date.loc[date.loc[:, column] == comb[0], 'score'],
                     date.loc[date.loc[:, column] == comb[1], 'score']).pvalue \
                <= 0.05 / len(combinations_all):  # Учли поправку Бонферони
            print('Найдены статистически значимые различия для колонки', column)
            break


for col in ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health',
            'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
            'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet',
            'romantic', 'absences']:
    get_stat_dif(student, col)
