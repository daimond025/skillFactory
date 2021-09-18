import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score




df = pd.read_csv('./data/spam7.csv')
mydict = {
    "y": 1,
    "n": 0,
}

df['spam'] = df['yesno'].apply(mydict.get)
df.drop(['yesno', 'Unnamed: 0'], axis=1, inplace=True)

X = df.drop(['spam'], axis=1)
y = df['spam']

random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)

model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,max_depth=3, min_samples_split=2,
                    min_samples_leaf=1, subsample=1,max_features=None, random_state=random_state)
model.fit(X_train, y_train)

Y_predicted = model.predict(X_test)
print(accuracy_score(y_test,Y_predicted))
imp_f = pd.Series(model.feature_importances_)
imp_f.sort_values(inplace = True)
print('3-й по важности признак: ',poly_X.iloc[:,[imp_f.index[-3]]].columns)