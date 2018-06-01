
import numpy as np
import xgboost as xgb

    
def prepare_data_model(dt, transf_vars, predictors_names, target_var):

    if transf_vars == 'to_matrix':
        
        dt = dt.as_matrix(columns = list(dt))
    
    elif transf_vars == 'to_DMatrix':     
        
       dt = xgb.DMatrix(dt[predictors_names].values,dt[target_var].values)
        
    return dt     


def fit_xgb_model(model, train, predictors_names, target_var, useTrainCV = False, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_params = model.get_xgb_params()
        transf_vars = 'to_DMatrix'
        xgtrain = prepare_data_model(train, transf_vars, predictors_names, target_var)
        cvresult = xgb.cv(xgb_params, xgtrain, num_boost_round = model.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          metrics='rmse', 
                          early_stopping_rounds = early_stopping_rounds, 
                          show_progress = True)
        model.set_params(n_estimators=cvresult.shape[0])
    
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
        train = dataset['train'],
        model_trained = fit_xgb_model(model, 
                                      train,
                                      predictors_names,
                                      target_var,
                                      useTrainCV = False, 
                                      cv_folds = 5, 
                                      early_stopping_rounds = 50)
    return model_trained