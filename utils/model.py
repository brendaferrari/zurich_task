from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from statistics import mean 
from statistics import stdev

class Model:

    def build_model(self, data_x, data_y, model, cross_val):
        r2=[]
        mae=[]
        mse=[]
        for train_index, test_index in cross_val.split(data_x, data_y):
            train_x, test_x = data_x.iloc[train_index], data_x.iloc[test_index]
            train_y, test_y = data_y.iloc[train_index], data_y.iloc[test_index]

            model.fit(train_x, train_y)
            test_pred_y = model.predict(test_x)
            r2.append(r2_score(test_y, test_pred_y))
            mae.append(mean_absolute_error(test_y, test_pred_y))
            mse.append(mean_squared_error(test_y, test_pred_y))
        print("R2: %0.2f (+/- %0.2f)\n MAE: %0.2f \n MSE: %0.2f" % (mean(r2), stdev(r2) * 2, mean(mae), mean(mse)))
