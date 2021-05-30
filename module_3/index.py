import re

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# pd.set_option('display.max_rows', None)

save_csv = 'main_task.csv'
date = pd.read_csv('main_task.xls')

# удаление дубликатов
date.drop_duplicates(subset=['ID_TA'], keep='first', inplace=True)

X = date.drop(['Restaurant_id'], axis=1)
y = date['Rating']

X['Number of Reviews'].fillna(X['Number of Reviews'].median(), inplace=True)

# TODO обработка URL_TA - извлекаем параметр - может пригодится
X['URL_TA']= X.apply(lambda row: str(row['URL_TA']).split('-')[1],axis=1)

# TODO  анализ по словам -- вводим новый признак по словам
def analisReviews(row):
    good_words = ['good', 'yummy', 'fine', 'great', 'tasty',
                  'satisfaction', 'amazing', 'nice', 'best',
                  'friendly', 'pleasant', 'exellent', 'loved',
                  'love', 'lovely', 'welcoming', 'wonderful',
                  'perfect', 'delicious', 'favourite', 'sweet',
                  'yum', 'adequate', 'happy', 'beautiful', 'liked', 'like']

    str_reviews = row['Reviews'].lower()
    count = 0
    for word in good_words:
        if word in str_reviews:
            count += 1
    row['Reviews_count'] = count
    return row
X = X.apply(lambda row: analisReviews(row), axis=1)

# TODO словарь по ценам
PriceRange = {'$$$$': 'very expensiv',
               '$$ - $$$': 'average',
               '$' : 'cheap'}
X['Price Range'] = X['Price Range'].map(PriceRange).fillna('Price unknown')

# dummy_price = pd.get_dummies(X['Price Range'])
# dummy_city = pd.get_dummies(X['City'])
# dummy_style = pd.get_dummies(X['Cuisine Style'])
# dummy_url = pd.get_dummies(X['URL_TA'])
# dummy_all = pd.concat([dummy_city, dummy_price], axis=1 )

# TODO обработка кухни
X['Cuisine Style'] = X['Cuisine Style'].str.replace('[', '', regex=False).str.replace(']', '', regex=False) \
    .str.replace("'", '', regex=False).str.strip()
X['Cuisine Style'] = X['Cuisine Style'].str.split(', ')
def lenCuisineStyle(style):
    if isinstance(style, list):
        return len(style)
    else:
        return 1
X['Cuisine_count'] = X['Cuisine Style'].apply(lenCuisineStyle)
X = X.explode('Cuisine Style')
X['Cuisine Style'].fillna('Cuisine unknown', inplace=True)


# TODO  обработка для даты отзывов
X['Reviews'] = X['Reviews'].str.replace('[[', '', regex=False).str.replace(']]', '', regex=False) \
    .str.replace('[', '', regex=False).str.replace("'", '', regex=False).str.strip()
X['Reviews'] = X['Reviews'].str.split('],')
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

    else:
        row['ReviewDate1'] = default
        row['ReviewDate2'] = default
    return row

X=  X.apply(lambda row: correctDate(row), axis=1)
X['ReviewDate1'] = pd.to_datetime(X['ReviewDate1'], format='%m/%d/%Y')
X['ReviewDate2'] = pd.to_datetime(X['ReviewDate2'], format='%m/%d/%Y')
X['ReviewDiffDate'] = ((X['ReviewDate2'] - X['ReviewDate1']).dt.days).abs()
X['ReviewDiffDate'].fillna(0, inplace=True)

X_save = X[['Cuisine_count', 'Number of Reviews', 'Reviews_count', 'ReviewDiffDate', 'Ranking', 'Rating']]

dummy_price = pd.get_dummies(X['Price Range'])
dummy_city = pd.get_dummies(X['City'])
dummy_style = pd.get_dummies(X['Cuisine Style'])
dummy_url = pd.get_dummies(X['URL_TA'])

# склейка массива
dummy_all = pd.concat([dummy_city, dummy_price,dummy_style, dummy_url, X_save ], axis=1)
dummy_all.to_csv(save_csv, index=False)
