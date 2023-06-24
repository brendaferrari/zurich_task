from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from statistics import mean, stdev
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *
import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from statistics import mean 
from sklearn.metrics import mean_squared_error

class Compare:

    def model_build_compare(self, X_data, Y_data, n_splits, seed, save_output=False):

        # Do a function to test number os splits https://stackoverflow.com/questions/42697551/scikit-learnpython-different-metric-resultsf1-score-for-stratifiedkfold

        #https://stackoverflow.com/questions/65318931/stratifiedkfold-vs-kfold-in-scikit-learn

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed) ## shuffle: divisão randomica, random state, split dependente, ver no doc, teste em loop
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)

        model_list = []
        accuracy_list = []
        balanced_accuracy_list = []
        f1_list = []
        time_list = []
        adj_r_squared_list = []
        r_squared_list = []
        rmse_list = []

        if Y_data.size == np.count_nonzero((Y_data=='active') | (Y_data=='inactive')):
            fold_no = 1
            
            for train_index, test_index in skf.split(X_data, Y_data):
                X_train_fold, X_test_fold = X_data.iloc[train_index], X_data.iloc[test_index]
                Y_train_fold, Y_test_fold = Y_data.iloc[train_index], Y_data.iloc[test_index]
                
                model, accuracy, balanced_accuracy, f1, time = self.model_test(X_train_fold, X_test_fold, Y_train_fold, Y_test_fold, classification=True, save_output=save_output)

                model_list.append(model)
                accuracy_list.append(accuracy)
                balanced_accuracy_list.append(balanced_accuracy)
                f1_list.append(f1)
                time_list.append(time)

                fold_no +=1

            balanced_accuracy_list_flattened  = [val for sublist in balanced_accuracy_list for val in sublist]
            accuracy_list_flattened  = [val for sublist in accuracy_list for val in sublist]
            f1_list_flattened  = [val for sublist in f1_list for val in sublist]

            maximum_acc = max(accuracy_list_flattened)
            index_acc = accuracy_list_flattened.index(maximum_acc)
            maximum_balacc = max(balanced_accuracy_list_flattened)
            index_balacc = balanced_accuracy_list_flattened.index(maximum_balacc)
            maximum_f1 = max(f1_list_flattened)
            index_f1 = f1_list_flattened.index(maximum_f1)

            print(f"\nMaximum Accuracy can be obtained by model {model_list[index_acc]} in split {index_acc}: {maximum_acc} %")
            print(f"\nStandard Deviation is: {stdev(accuracy_list_flattened)}")
            print(f"\nMaximum Balanced Accuracy can be obtained by model {model_list[index_balacc]} in split {index_balacc}: {maximum_balacc} %")
            print(f"\nStandard Deviation is: {stdev(balanced_accuracy_list_flattened)}")
            print(f"\nMaximum F1 can be obtained by model {model_list[index_f1]} in split {index_f1}: {maximum_f1} %")
            print(f"\nStandard Deviation is: {stdev(f1_list_flattened)}")
        else:
            fold_no = 1
            for train_index, test_index in kf.split(X_data, Y_data):
                X_train_fold, X_test_fold = X_data.iloc[train_index], X_data.iloc[test_index]
                Y_train_fold, Y_test_fold = Y_data.iloc[train_index], Y_data.iloc[test_index]

                model, adj_r_squared, r_squared, rmse, time = self.model_test(X_train_fold, X_test_fold, Y_train_fold, Y_test_fold, classification=False, save_output=save_output)

                model_list.append(model)
                adj_r_squared_list.append(adj_r_squared)
                r_squared_list.append(r_squared)
                rmse_list.append(rmse)
                time_list.append(time)

                fold_no +=1

            adj_r_squared_list_flattened  = [val for sublist in adj_r_squared_list for val in sublist]
            r_squared_list_flattened  = [val for sublist in r_squared_list for val in sublist]
            rmse_list_flattened  = [val for sublist in rmse_list for val in sublist]

            maximum_ars = max(adj_r_squared_list_flattened)
            index_ars = adj_r_squared_list_flattened.index(maximum_ars)
            maximum_rs = max(r_squared_list_flattened)
            index_rs = r_squared_list_flattened.index(maximum_rs)
            maximum_rmse = min(rmse_list_flattened)
            index_rmse = rmse_list_flattened.index(maximum_rmse)

            print(f"\nMaximum Ajusted R-squared can be obtained by model {model_list[index_ars]} in split {index_ars}: {maximum_ars} %")
            print(f"\nStandard Deviation is: {stdev(adj_r_squared_list_flattened)}")
            print(f"\nMaximum R-squared can be obtained by model {model_list[index_rs]} in split {index_rs}: {maximum_rs} %")
            print(f"\nStandard Deviation is: {stdev(r_squared_list_flattened)}")
            print(f"\nMin RMSE can be obtained by model {model_list[index_rmse]} in split {index_rmse}: {maximum_rmse} %")
            print(f"\nStandard Deviation is: {stdev(rmse_list_flattened)}")

        return

    def model_test(self, X_train_fold, X_test_fold, Y_train_fold, Y_test_fold, classification=False, save_output=False):

        model_name = []
        accuracy = []
        balanced_accuracy = []
        f1 = []
        time = []
        adj_r_squared = []
        r_squared = []
        rmse = []

        if classification==True:
            model = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
            train,test = model.fit(X_train_fold, X_test_fold, Y_train_fold, Y_test_fold)

            result = train.reset_index()

            print(f"For {result.iloc[0,0]} acc: {result.iloc[0,1]} bal_acc: {result.iloc[0,2]}, f1: {result.iloc[0,4]}, time: {result.iloc[0,5]}")
            print(f"For {result.iloc[1,0]} acc: {result.iloc[1,1]} bal_acc: {result.iloc[1,2]}, f1: {result.iloc[1,4]}, time: {result.iloc[1,5]}")
            print(f"For {result.iloc[2,0]} acc: {result.iloc[2,1]} bal_acc: {result.iloc[2,2]}, f1: {result.iloc[2,4]}, time: {result.iloc[2,5]}")

            model_name.append(result.iloc[0,0])
            accuracy.append(result.iloc[0,1])
            balanced_accuracy.append(result.iloc[0,2])
            f1.append(result.iloc[0,4])
            time.append(result.iloc[0,5])
        else:
            model = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)
            train,test = model.fit(X_train_fold, X_test_fold, Y_train_fold, Y_test_fold)

            result = train.reset_index()

            print(f"For {result.iloc[0,0]} adj_r_sq: {result.iloc[0,1]} r_sq: {result.iloc[0,2]}, rmse: {result.iloc[0,3]}, time: {result.iloc[0,4]}")
            print(f"For {result.iloc[1,0]} adj_r_sq: {result.iloc[1,1]} r_sq: {result.iloc[1,2]}, rmse: {result.iloc[1,3]}, time: {result.iloc[1,4]}")
            print(f"For {result.iloc[2,0]} adj_r_sq: {result.iloc[2,1]} r_sq: {result.iloc[2,2]}, rmse: {result.iloc[2,3]}, time: {result.iloc[2,4]}")

            model_name.append(result.iloc[0,0])
            adj_r_squared.append(result.iloc[0,1])
            r_squared.append(result.iloc[0,2])
            rmse.append(result.iloc[0,3])
            time.append(result.iloc[0,4])

        if save_output == True:
            train.to_csv(f'train.csv')
            test.to_csv(f'test.csv')

        if classification==True:
            return (model_name, accuracy, balanced_accuracy, f1, time)
        else:
            return (model_name, adj_r_squared, r_squared, rmse, time)


    def hyperparameter_search(self, x_train, y_train, reg, param_grid):
        # reg = {"linear_regression": LinearRegression(),"sgd_regressor": SGDRegressor(),
        # "gradient_boosting_regressor": GradientBoostingRegressor(), "elastic_net": ElasticNet(),
        #     "bayesian_ridge": linear_model.BayesianRidge(), "kernel_ridge": KernelRidge(),
        #     "svr": SVR(), "nusvr": NuSVR(), "linear_svr": LinearSVR()
        #     }
            
        # param_grid = {"linear_regression": {},
        #             "sgd_regressor": {"loss": ["huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
        #             "penalty": ["l2", "l1", "elasticnet", None], "alpha": [1, 0.1, 0.01, 0.001, 0.0001]},
        #             "gradient_boosting_regressor": {"loss": ["huber", "quantile"],
        #             "learning_rate": [1, 0.1, 0.01, 0.001], "n_estimators": [100,1000]},
        #             "elastic_net": {"alpha": [1, 0.1, 0.01, 0.001, 0.0001], "l1_ratio": [0.2,0.5,0.75]},
        #             "bayesian_ridge": {"n_iter": [100,1000,10000]}, "kernel_ridge": {"alpha": [1],
        #             "kernel": ["linear", "poly", "rbf", "sigmoid"], "degree": [2,3,4,5,6]}, 
        #             "svr": {"kernel": ["linear", "poly", "rbf", "sigmoid"], "gamma":
        #             ["scale", "auto"], "epsilon": [0.1, 1, 100], "C": [1, 100, 1000]}, 
        #             "nusvr": {"nu": [0.25, 0.5, 0.75], 
        #             "kernel": ["linear", "poly", "rbf", "sigmoid"], "gamma":
        #             ["scale", "auto"], "C": [1, 100, 1000]}, "linear_svr": {"epsilon": [0.1, 1, 100], "loss": 
        #             ["epsilon_insensitive", "squared_epsilon_insensitive"]}}

        best_estimator = []
        best_score = []
        if y_train.size == np.count_nonzero((y_train=='active') | (y_train=='inactive')):
            for key, value in reg.items():
                    #print(f"Hyperparameter for {key}")
                    #try:
                            grid_search = GridSearchCV(value, param_grid.get(key), cv=3, refit='accuracy', scoring='accuracy', return_train_score=True, verbose=10)
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
                            grid_search = GridSearchCV(value, param_grid.get(key), cv=3, refit='r2', scoring=['r2', 'neg_mean_squared_error'], return_train_score=True, verbose=10)
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
                print("R2: %0.2f (+/- %0.2f)\n Jtrain: %0.2f \n Jtest: %0.2f" % (score.mean(), score.std() * 2, mean(train_error), mean(test_error)))
        
        results = pd.DataFrame([scores, Jtrain, Jtest], columns=pd.MultiIndex.from_tuples(header), index=['R2', 'Jtrain', 'Jtest'])

        return results