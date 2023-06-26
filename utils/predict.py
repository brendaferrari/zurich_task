from sklearn.metrics import r2_score

class Predict:

    def model_prediction(self, data_x, data_y, models, cross_val, predict_data):     
        
        r2 = 0
        pred_y_actual = []
        test_y_pred = []
        test_y_actual = []
        for train_index, test_index in cross_val.split(data_x, data_y):
            train_x, test_x = data_x.iloc[train_index], data_x.iloc[test_index]
            train_y, test_y = data_y.iloc[train_index], data_y.iloc[test_index]

            models.fit(train_x, train_y)
            test_pred_y = models.predict(test_x)
            pred_y_ = models.predict(predict_data)
            r2_score_ = r2_score(test_y, test_pred_y)
            print(f"For this iteration -> Cross Validation score: {r2_score}")
            if r2 < r2_score_:
                r2 = r2_score_
                test_y_pred = test_pred_y
                pred_y_actual = pred_y_
                test_y_actual = test_y
        return pred_y_actual, test_y_pred, test_y_actual
    
    def write_results(self, array, path):

        with open(path, mode='w') as out:
            out.write(f'prediction\n')
            for result in list(array):
                out.write(f'{result}\n')