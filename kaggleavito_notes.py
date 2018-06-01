
# Create Validation Index and Remove Dead Variables
training_index = df.loc[df.activation_date <= pd.to_datetime('2017-04-07')].index
validation_index = df.loc[df.activation_date >= pd.to_datetime('2017-04-08')].index

print("the training_index is :", training_index)
print("the validation_index is :", validation_index)

# we change set to validation_index
df.loc[df['activation_date'] >= pd.to_datetime('2017-04-08') & ]


model = xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)

#Missing values can be replaced by a default value in the DMatrix constructor:
#Weights can be set when needed:
train_X = xgb.DMatrix(data, label=label, missing=-999.0, weight = w)

#Saving DMatrix into a XGBoost binary file will make loading faster:
train_X.save_binary('train.buffer')

# Setting Parameters
   # General Parameters
param = {'silent': 0, 'booster': 'gbtree'}
# booster: if gbtree doesn't work, try: 'booster': 'gblinear
# nthread: default to maximum

# Booster Parameters
param['eta'] = 0.1 #[0,1], [default=0.3], typical values: 0.01-0.2

# ------------ Control Overfitting ---------------
param['max_depth'] =  5 # **CV**, [3..10],[default = 6], 0 minims there is no limit
# max_depth: Higher values prevent a model from learning relations very specific to a particular sample
param['gamma'] = 0 # [default = 0]
# gamma: minimum loss reduction required to make a further partition on a leaf node of the three.
# The larger, the more conservative the algorithm will be
# Its value will depend largely on the loss function so it should be tuned
param['min_child_weight'] = 1 #**CV** [default = 1]
# min_child_weight: Minimum sum of weigths of all observations required in a child
# higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree
# Too high values can lead to under-fitting, hence it should be tuned using CV!
param['subsample'] = 1  # [default = 1], typical values = [0.5 -1]
#subsample: subsample ratio of the training instance [0-1]
#Setting it to 0.5 means that XGBoost randomly collected half of the data instances to grow trees and this will prevent overfitting.
param['colsample_bytree'] = 1 # [default = 1], typical values = [0.5 -1]
# Fraction of columns to be randomly samples for each tree.
param['lambda'] = 1 # [default = 1]
#L2 regularization term on weights (analogous to Ridge regression)
#This used to handle the regularization part of XGBoost. Many data scientists don’t use it often.
# -----------------------------------------------

# ------------ Control Class Imbalance ---------------
param['max_delta_step'] = 0 #default = 0, [1-10]
#Maximum delta step we allow each tree’s weight estimation to be.
#If the value is set to 0, it means there is no constraint.
#If it is set to a positive value, it can help making the update step more conservative.
#Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
param['scale_pos_weight '] = 1 # default = 1
#A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.
#Use stratified cross validation to enforce class distributions
# when there are a large number of classes or an imbalance in instances for each class.
# ----------------------------------------------------

#------------------- Control Speed -------------------
param['alpha'] = 0 # default = 0
# L1 regularization term on weight (analogous to Lasso regression)
# Can be used in case of very high dimensionality so that the algorithm runs faster when implemented
# ----------------------------------------------------

   # Learning Task Parameters
param['objective'] = 'reg:linear'
param['eval_metric'] = 'rmse'
param['seed'] = 0 # default to 0,  can be used for generating reproducible results and also for parameter tuning
#base_score [default=0.5] the initial prediction score of all instances, global bias
#For sufficient number of iterations, changing this value will not have too much effect.




# Training
#Specify validations set to watch performance
evallist = [(val_X, 'validation'), (train_X, 'train')]
num_round = 10







param = {'eval_metric': '','objective': 'reg:linear'}

   # Booster Parameters
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 4
param['eval_metric'] = 'auc'

#QUITAR LA SEED DE VALIDATION SET !!!
