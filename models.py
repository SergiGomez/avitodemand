
import numpy as np
import xgboost as xgb
import sklearn as sk
from sklearn.model_selection import KFold

def prepare_data_model(dt, transf_vars, predictors_names, target_var):

    if transf_vars == 'to_matrix':

        dt = dt.as_matrix(columns = list(dt))

    elif transf_vars == 'to_DMatrix':

       dt = xgb.DMatrix(dt[predictors_names].values,dt[target_var].values)

    return dt


def fit_xgb_model(model, train, predictors_names, target_var, useTrainCV = 'None', cv_folds=5, early_stopping_rounds=50):

    if useTrainCV == "cv_xgb":
        xgb_params = model.get_xgb_params()
        transf_vars = 'to_DMatrix'
        xgtrain = prepare_data_model(train, transf_vars, predictors_names, target_var)
        cvresult = xgb.cv(xgb_params, xgtrain, num_boost_round = model.get_params()['n_estimators'],
                          nfold=cv_folds,
                          metrics='rmse',
                          early_stopping_rounds = early_stopping_rounds,
                          show_progress = True)
        model.set_params(n_estimators=cvresult.shape[0])

    elif useTrainCV == "cv_sk":
        X = train[predictors_names]
        y = train[target_var]
        kf = RepeatedKFold(n_splits = cv_folds, n_repeats = cv_folds, random_state = None)
        for train_index, dev_index in kf.split(X):
            X_train, X_dev = X[train_index], X[dev_index]
            y_train, y_dev = y[train_index], y[dev_index]
            print("Train:", train_index, "Validation:",dev_index)
            model = xgb.XGBRegressor()
            model.set_params(**params)
            model.fit(train[predictors_names], train[target_var], eval_metric='rmse')

    #Fit the algorithm on the data
    model.fit(train[predictors_names], train[target_var], eval_metric='rmse')

    return model


def create_model(dataset, model_name, target_var, params):

    # Predictors_names
    predictors_names = [x for x in dataset['train'].columns if x not in [target_var]]

    if model_name == 'xgb':
        #dataset = prepare_data_model(model_name, dataset)
        model = xgb.XGBRegressor()
        model.set_params(**params)
        train = dataset['train']
        model_trained = fit_xgb_model(model = model,
                                      train = train,
                                      predictors_names = predictors_names,
                                      target_var = target_var,
                                      useTrainCV = True,
                                      cv_folds = 5,
                                      early_stopping_rounds = 50)
    return model_trained
