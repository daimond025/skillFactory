
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
