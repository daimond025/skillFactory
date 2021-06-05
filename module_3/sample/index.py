import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import statistics

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import os


class PrepareDate():
    # в констуркоре определим методы обработки данных
    def __init__(self, df_output):
        self.df_output = df_output

    def processData(self):
        df_output = self.df_output.copy()

        df_output.drop(['ID_TA', ], axis=1, inplace=True)
        # признак сети - одинаковые Restaurant_id для ресторанов
        # УБИРАЕМ - так как новый признак никак не повлиял на результаьт в результате опыта
        # df_output = self.netRestaurant(df_output)

        # Далее заполняем пропуски
        df_output['Number of Reviews'].fillna(0, inplace=True)

        # нормировка значения Ranking
        df_output = self.normilizeRanking(df_output)

        # обработка цен - заполяняем цены согласно словарю
        df_output = self.ProcessPrice(df_output)
        #  обрботка  кухни - вводим новый признак (количество кухонь) + распарсиваем строку кухонь
        df_output = self.ProcessCuisine(df_output)

        # url - обработка url - извлекаем параметр
        df_output = self.ProcessURL(df_output)

        #  обрботка отзывов - вводим новые признаки:
        #  - средняя длина отзыва
        #  - количество слов (превосходные отзывы - поиск по словам)
        #  - взлечени дат - вычисдение разницы между датами отыва
        # Убираем эти параметры - большое время обработки и малый вклад в точность прогнозирования (по результатам опыта)
        # df_output = self.ProcessReviews(df_output)

        # вводим переменные dummi как города так и стили кухонь - такое действие позволяет не выводить новые признаки в
        # каких городах пицерии/бары/кафетерии и т.п. !!!
        df_output = pd.get_dummies(df_output, columns=['City', 'Cuisine Style', 'URL_TA'], dummy_na=True)

        # убираем не нужные для модели признаки
        object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object' and s !='Restaurant_id' ]
        df_output.drop(object_columns, axis=1, inplace=True)

        return df_output

    def netRestaurant(self, data):
        list_net = (data['Restaurant_id'].value_counts()[data['Restaurant_id'].value_counts() > 1].index)
        data['net'] = data.apply(lambda row: 1 if row['Restaurant_id'] in list_net else 0, axis=1)
        return data


    def ProcessURL(self, data):
        data['URL_TA'] = data.apply(lambda row: str(row['URL_TA']).split('-')[1], axis=1)
        return data

    def ProcessPrice(self, data):
        PriceRange = {
            '$$$$': 3,
            '$$ - $$$': 2,
            '$': 1
        }
        data['Price Range'] = data['Price Range'].map(PriceRange).fillna(0)
        return data

    def ProcessReviews(self, data):
        data['Reviews'] = data['Reviews'].str.replace('[[', '', regex=False).str.replace(']]', '', regex=False) \
            .str.replace('[', '', regex=False).str.replace("'", '', regex=False).str.strip()
        data['Reviews'] = data['Reviews'].str.split('],')

        def correctDate(row):
            default = '01/01/2000'

            if isinstance(row['Reviews'], list):
                if len(row['Reviews']) == 2:
                    dates = str(row['Reviews'][1]).replace(' ', '')
                    Reviews_arr = dates.split(',')
                    if len(Reviews_arr) == 2:
                        row['ReviewDate1'] = Reviews_arr[0]
                        row['ReviewDate2'] = Reviews_arr[1]
                    elif len(Reviews_arr) == 1 and Reviews_arr[0] != '':
                        row['ReviewDate1'] = Reviews_arr[0]
                        row['ReviewDate2'] = Reviews_arr[0]
                    else:
                        row['ReviewDate1'] = default
                        row['ReviewDate2'] = default

                    # среднее количество слов
                    Reviews_str = str(row['Reviews'][0])
                    Reviews_str_arr = Reviews_str.split(',')
                    if len(Reviews_str_arr) == 2:
                        l = [len(Reviews_str_arr[0]), len(Reviews_str_arr[1])]
                        row['ReviewLen'] = statistics.mean(l)
                    elif len(Reviews_str_arr) == 1 and Reviews_str_arr[0] != '':
                        l = [len(Reviews_str_arr[0])]
                        row['ReviewLen'] = statistics.mean(l)
                    else:
                        row['ReviewLen'] = 0

                    #  по словам
                    good_words = ['good', 'yummy', 'fine', 'great', 'tasty',
                                  'satisfaction', 'amazing', 'nice', 'best',
                                  'friendly', 'pleasant', 'exellent', 'loved',
                                  'love', 'lovely', 'welcoming', 'wonderful',
                                  'perfect', 'delicious', 'favourite', 'sweet',
                                  'yum', 'adequate', 'happy', 'beautiful', 'liked', 'like']

                    str_reviews = str(row['Reviews'][0]).lower()
                    count = 0
                    for word in good_words:
                        if word in str_reviews:
                            count += 1
                    row['Reviews_count'] = count
            else:
                row['ReviewDate1'] = default
                row['ReviewDate2'] = default
                row['ReviewLen'] = 0
                row['Reviews_count'] = 0
            return row

        data = data.apply(lambda row: correctDate(row), axis=1)
        data['ReviewDate1'] = pd.to_datetime(data['ReviewDate1'], format='%m/%d/%Y')
        data['ReviewDate2'] = pd.to_datetime(data['ReviewDate2'], format='%m/%d/%Y')
        data['ReviewDiffDate'] = ((data['ReviewDate2'] - data['ReviewDate1']).dt.days).abs()
        data['ReviewDiffDate'].fillna(0, inplace=True)
        data['ReviewLen'].fillna(0, inplace=True)
        data['Reviews_count'].fillna(0, inplace=True)
        data = data.drop(['ReviewDate1', 'ReviewDate2'], axis=1)

        return data

    def ProcessCuisine(self, data):
        # TODO обработка кухни
        data['Cuisine Style'] = data['Cuisine Style'].str.replace('[', '', regex=False).str.replace(']', '',
                                                                                                    regex=False) \
            .str.replace("'", '', regex=False).str.strip()
        data['Cuisine Style'] = data['Cuisine Style'].str.split(', ')

        data['Cuisine_count'] = data['Cuisine Style'].apply(lambda X: len(X) if isinstance(X, list) else 1)

        data = data.explode('Cuisine Style')
        return data

    def normilizeRanking(self, data):
        for x in (data['City'].value_counts()).index:
            temp = data['Ranking'][data['City'] == x].copy()
            min = temp.min()
            max = temp.max()
            data.loc[data['City'] == x, 'Ranking'] = (data['Ranking'] - min) / (max - min)
        return data


for dirname, _, filenames in os.walk(os.getcwd() + '/kaggle/input'):

    for filename in filenames:
        print(os.path.join(dirname, filename))

RANDOM_SEED = 42

DATA_DIR = os.getcwd() + '/kaggle/input'
df_train = pd.read_csv(DATA_DIR + '/main_task.csv')
df_test = pd.read_csv(DATA_DIR + '/kaggle_task.csv')
sample_submission = pd.read_csv(DATA_DIR + '/sample_submission.csv')

df_train['sample'] = 1
df_test['sample'] = 0
df_test['Rating'] = 0
data = df_test.append(df_train, sort=False).reset_index(drop=True)

DATA_DIR = os.getcwd() + '/kaggle/input'
df_train = pd.read_csv(DATA_DIR + '/main_task.csv')
df_test = pd.read_csv(DATA_DIR + '/kaggle_task.csv')
df_train['sample'] = 1
df_test['sample'] = 0
df_test['Rating'] = 0


data = df_test.append(df_train, sort=False).reset_index(drop=True)


df_object = PrepareDate(data)
df_preproc = df_object.processData()

train_data = df_preproc.query('sample == 1').drop(['sample', 'Restaurant_id'], axis=1)
# подготовка тестируемых данных - мы расширили списко по параметры стилей кухонь!!!
test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)
test_data = test_data.drop_duplicates(subset=['Restaurant_id'], keep='first').drop(['Restaurant_id'], axis=1)



y = train_data.Rating.values
X = train_data.drop(['Rating'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))


print(test_data.columns)
exit()

test_data = test_data.drop(['Rating'], axis=1)
predict_submission = model.predict(test_data)
sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)

# print('MAE2:', metrics.mean_absolute_error(y_pred_1, predict_submission))
