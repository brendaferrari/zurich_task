from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from statistics import mean 
from statistics import stdev

class Predict:

    def model_prediction(self, data_x, data_y, model, cross_val, predict_data):

        for train_index, test_index in cross_val.split(data_x, data_y):
            train_x, test_x = data_x.iloc[train_index], data_x.iloc[test_index]
            train_y, test_y = data_y.iloc[train_index], data_y.iloc[test_index]

            model.fit(train_x, train_y)
            test_pred_y = model.predict(predict_data)
        return test_pred_y