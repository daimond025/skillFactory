import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

date = pd.read_csv('main_task.xls')

X = date.drop(['Restaurant_id', 'Rating'], axis=1)
y = date['Rating']

# обработка для даты отхывов
X['Reviews'] = X['Reviews'].str.replace('[', '', regex=False).str.replace(']', '', regex=False) \
    .str.replace("'", '', regex=False).str.strip()
X['Reviews'] = X['Reviews'].str.split(', ')

def review_formated(row):
    if len(row['Reviews']) != 4:
        row['ReviewDate1'] = None
        row['ReviewDate2'] = None
    else:
        row['ReviewDate1'] = row['Reviews'][2]
        row['ReviewDate2'] = row['Reviews'][3]
    return row

X = X.apply(review_formated, axis=1)
print(X.iloc[0])
exit()

#  обработка для стиля кухни
X['Cuisine Style'] = X['Cuisine Style'].str.replace('[', '', regex=False).str.replace(']', '', regex=False) \
    .str.replace("'", '', regex=False).str.strip()
X['Cuisine Style'] = X['Cuisine Style'].str.split(', ')


def lenCuisineStyle(style):
    if isinstance(style, list):
        return len(style)
    else:
        return 1


X['Cuisine Style arr'] = X['Cuisine Style'].apply(lenCuisineStyle)
X = X.explode('Cuisine Style')

#  обработка данных
X['Number of Reviews'].fillna(X['Number of Reviews'].median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
regr = RandomForestRegressor(n_estimators=100)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
