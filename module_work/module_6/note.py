
#  TODO полинальные своства - обучение + какие свойства больше влияют
pf = PolynomialFeatures(interaction_only=True, include_bias=False)
poly_data = pf.fit_transform(X)
poly_cols = pf.get_feature_names(X.columns)
poly_cols = [x.replace(' ', '_') for x in poly_cols]

poly_X = pd.DataFrame(poly_data, columns=poly_cols)


random_state=42
X_train, X_test, y_train, y_test = train_test_split(poly_X, y, test_size=0.20, random_state=random_state)

# model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100,max_depth=3, min_samples_split=2,
#                     min_samples_leaf=1, subsample=1,max_features=None, random_state=random_state)
# model.fit(X_train, y_train)
# Y_predicted = model.predict(X_test)
# print(accuracy_score(y_test,Y_predicted))


# imp_f = pd.Series(model.feature_importances_)
# imp_f.index = poly_cols
# imp_f.sort_values(inplace = True)
# imp_f.plot(kind = 'barh')



# TODO GridSearchCV - перебор параметров модели
#
# param_grid = {'learning_rate':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
#               'n_estimators':[100, 250, 500, 750, 1000, 1250, 1500, 1750]}
# model =  GradientBoostingClassifier(random_state=random_state)
# clf = GridSearchCV(model, param_grid, scoring='accuracy', n_jobs=-1, cv=5)
# clf.fit(X_train, y_train)
# print(clf.best_params_)
# print(clf.best_score_)


# Создаем словать с кодировкой значений в цвета
color_dict = {'040001': 'чёрный', 'EE1D19': 'красный', '0000CC': 'синий', 'CACECB': 'серебристый', '007F00': 'зелёный',
              'FAFBFB': 'белый', '97948F': 'серый', '22A0F8': 'голубой', '660099': 'пурпурный', '200204': 'коричневый',
              'C49648': 'бежевый', 'DEA522': 'золотистый', '4A2197': 'фиолетовый', 'FFD600': 'жёлтый',
              'FF8649': 'оранжевый', 'FFC0CB': 'розовый'}

data['color'].replace(to_replace=color_dict, inplace=True)


{'EUROPEAN': ['BMW', 'VOLKSWAGEN', 'MERCEDES', 'AUDI', 'SKODA', 'VOLVO'],
 'JAPANESE': ['NISSAN', 'TOYOTA', 'MITSUBISHI', 'HONDA', 'INFINITI', 'LEXUS']}

eur_append = ['PORSCHE', 'LAND_ROVER', 'JAGUAR',
              'MINI', 'RENAULT', 'OPEL', 'PEUGEOT', 'CITROEN']
jap_append = ['SUBARU', 'MAZDA', 'SUZUKI']

#  url cars
url_cars = [
   'https://auto.ru/cars/all/?page=',  # ALL
   'https://auto.ru/cars/vaz/all/?page=',  # BAZ
   'https://auto.ru/cars/audi/all/?page=',  # audi
   'https://auto.ru/cars/bmw/all/?page=',  # bmw
   'https://auto.ru/cars/chery/all/?page=',  # chery
   'https://auto.ru/cars/chevrolet/all/?page=',  # chevrolet
   'https://auto.ru/cars/citroen/all/?page=',  # citroen
   'https://auto.ru/cars/daewoo/all/?page=',  # daewoo
   'https://auto.ru/cars/ford/all/?page=',  # ford
   'https://auto.ru/cars/geely/all/?page=',  # geely
   'https://auto.ru/cars/honda/all/?page=',  # honda
   'https://auto.ru/cars/hyundai/all/?page=',  # hyundai
   'https://auto.ru/cars/infiniti/all/?page=',  # infiniti
   'https://auto.ru/cars/kia/all/?page=',  # kia
   'https://auto.ru/cars/land_rover/all/?page=',  # land_rover
   'https://auto.ru/cars/lexus/all/?page=',  # lexus
   'https://auto.ru/cars/mazda/all/?page=',  # mazda
   'https://auto.ru/cars/mercedes/all/?page=',  # mercedes
   'https://auto.ru/cars/mitsubishi/all/?page=',  # mitsubishi
   'https://auto.ru/cars/nissan/all/?page=',  # nissan
   'https://auto.ru/cars/opel/all/?page=',  # opel
   'https://auto.ru/cars/peugeot/all/?page=',  # peugeot
   'https://auto.ru/cars/porsche/all/?page=',  # porsche
   'https://auto.ru/cars/renault/all/?page=',  # renault
   'https://auto.ru/cars/skoda/all/?page=',  # skoda
   'https://auto.ru/cars/subaru/all/?page=',  # subaru
   'https://auto.ru/cars/suzuki/all/?page=',  # suzuki
   'https://auto.ru/cars/toyota/all/?page=',  # toyota
   'https://auto.ru/cars/volkswagen/all/?page=',  # volkswagen
   'https://auto.ru/cars/gaz/all/?page=',  # gaz
]

# TODO EXAMPLE
# https://www.kaggle.com/juliagil/sf-dst-car-price-prediction-dspr-28-team#-3.-EDA-AND-BASIC-DATA-CLEANING