
import numpy as np
import xgboost as xgb

##################### XGBOOST #######################

if model_name == 'xgb':
	#Saving DMatrix into a XGBoost binary file will make loading faster:
	#train_X.save_binary('train.buffer')
	#Definition of parameters
	param = {'silent': 0, 'booster': 'gbtree', 'objective': 'reg:linear'}
	param['eta'] = 0.1
	param['max_depth'] =  5
	param['gamma'] = 0
	param['min_child_weight'] = 1
	param['subsample'] = 1
	param['colsample_bytree'] = 1
	param['lambda'] = 1
	param['max_delta_step'] = 0
	param['scale_pos_weight '] = 1
	param['alpha'] = 0

	print('Running cross validation')
	# Do cross validation, this will print result out as
	# [iteration]  metric_name:mean_value+std_value
	# std_value is standard deviation of the metric

	xgTrain = xgb.DMatrix(train_X, label = train_Y)
	xgVal = xgb.DMatrix(val_X, label = val_Y)

	res = xgb.cv(param, train_X, num_boost_round = 10, nfold = 10,
		   metrics={'rmse'}, seed = 0,
		   callbacks=[xgb.callback.print_evaluation(show_stdv=True)])
	# xgb.cv(param, dtrain, num_round, nfold=5,
		   # metrics={'error'}, seed=0,
		   # callbacks=[xgb.callback.print_evaluation(show_stdv=True),
		   # xgb.callback.early_stop(3)])
	print(res, verbose = True)

	#Training
	model = xgb.train(param, xgTrain, num_round)
