from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from statistics import mean
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from statistics import mean 
from sklearn.metrics import mean_squared_error

class Compare:

    def hyperparameter_search(self, x_train, y_train, reg, param_grid):

        best_estimator = []
        best_score = []
        if y_train.size == np.count_nonzero((y_train=='active') | (y_train=='inactive')):
            for key, value in reg.items():
                    #print(f"Hyperparameter for {key}")
                    #try:
                            grid_search = GridSearchCV(value, param_grid.get(key), cv=3, refit='accuracy', scoring='accuracy', return_train_score=True, verbose=25)
                            grid_search.fit(x_train, y_train.to_numpy().ravel())
                            best_estimator.append(grid_search.best_estimator_)
                            best_score.append(grid_search.best_score_)
                            print(grid_search.best_estimator_)
                    #except(ValueError):
                            #print('not calculated')

        else:
            for key, value in reg.items():
                    #print(f"Hyperparameter for {key}")
                    #try:
                            grid_search = GridSearchCV(value, param_grid.get(key), cv=3, refit='r2', scoring=['r2', 'neg_mean_squared_error'], return_train_score=True, verbose=25)
                            grid_search.fit(x_train, y_train.to_numpy().ravel())
                            best_estimator.append(grid_search.best_estimator_)
                            best_score.append(grid_search.best_score_)
                            print(grid_search.best_estimator_)
                    #except(ValueError):
                            #print('not calculated')

        return best_estimator, best_score
    
    def compare_models(self, models, datasets):
        
        header = []
        scores = []
        Jtrain = []
        Jtest = []

        for name, data in datasets.items():
            train_error = []
            test_error = []
            for model in models:
                train_error = []
                test_error = []
                kf = KFold(n_splits=5, random_state=42, shuffle=True)
                for train_index, test_index in kf.split(data[0], data[1]):
                    train_x, test_x = data[0].iloc[train_index], data[0].iloc[test_index]
                    train_y, test_y = data[1].iloc[train_index], data[1].iloc[test_index]

                    model.fit(train_x, train_y)
                    train_pred_y = model.predict(train_x)
                    test_pred_y = model.predict(test_x)
                    train_error.append(mean_squared_error(train_y, train_pred_y))
                    test_error.append(mean_squared_error(test_y, test_pred_y))

                score = cross_val_score(model, data[0], data[1], cv=5, scoring='r2')
                scores.append(score.mean())
                Jtrain.append(mean(train_error))
                Jtest.append(mean(test_error))
                header.append((name,str(model)))
                print(f"Metrics for {name} using {model} -> Cross Validation score: {score.mean():.2f} (+/- {score.std() * 2:.2f}),\
                      Jtrain: {mean(train_error):.2f}, Jtest: {mean(test_error):.2f}")
        
        results = pd.DataFrame([scores, Jtrain, Jtest], columns=pd.MultiIndex.from_tuples(header), index=['CV_score', 'Jtrain', 'Jtest'])

        return results