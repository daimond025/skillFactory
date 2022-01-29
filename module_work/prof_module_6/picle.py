from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import numpy as np
import pickle

X, y = load_diabetes(return_X_y=True)
X = X[:, 0].reshape(-1, 1) # Берём только один признак
regressor = LinearRegression()
regressor.fit(X,y)

value_to_predict = np.array([0]).reshape(-1, 1)
print(regressor.predict(value_to_predict))

with open('./myfile.pkl', 'wb') as output:
   	pickle.dump(regressor, output) #Сохраняем

with open('./myfile.pkl', 'rb') as pkl_file:
    	regressor_from_file = pickle.load(pkl_file) #Загружаем

print(regressor_from_file.predict(value_to_predict))