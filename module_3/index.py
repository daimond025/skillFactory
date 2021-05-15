
import  pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

date = pd.read_csv('main_task.xls')

X = date.drop(['Restaurant_id', 'Rating'], axis=1)
y = date['Rating']

print(X.iloc[0]['Reviews'])
exit()
print(X['Price Range'].unique())
print(X['City'].nunique())


X['Cuisine Style'] = X['Cuisine Style'].str.replace('[', '',regex=False).str.replace(']', '',regex=False)\
    .str.replace("'", '',regex=False).str.strip()


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

