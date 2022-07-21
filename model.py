import pandas as pd
import numpy as np
import sklearn
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

housing=pd.read_csv('data.csv')

#Test and Train Splitting using sklearn
train_set, test_test=train_test_split(housing, random_state=42,test_size=0.2)

##stratified train test should be done so that test and train datas show good representation in both test and train
from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
housing=strat_train_set

## separating labels and features
housing=strat_train_set.drop('MEDV',axis=1)
housing_labels=strat_train_set['MEDV'].copy()

##Pipelining

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline=Pipeline([('numpy',SimpleImputer(strategy='median')),('std_scaler',StandardScaler())])
# rather than applying all of them individually you can just make a pipeline and apply all this at once on the trainset of housing data
housing_num_tr=my_pipeline.fit_transform(housing) #this is an array
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()
#model=DecisionTreeRegressor()
#model=LinearRegression()
model.fit(housing_num_tr,housing_labels)
 

some_data=housing.iloc[:5]
some_labels=housing_labels.iloc[:5]
prep_data=my_pipeline.transform(some_data)

## Evaluating the model
from sklearn.metrics import mean_squared_error
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_labels, housing_predictions)
rmse=np.sqrt(mse)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer_names
scores=cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores=np.sqrt(-scores)
#print(rmse_scores)
# def print_scores():
#     print("Scores: ",rmse_scores)
#     print("Mean: ",rmse_scores.mean())
#     print("Deviation: ", rmse_scores.std())
# print_scores()

# print("Testing the data:")
X_test=strat_test_set.drop('MEDV',axis=1)
model_columns=list(X_test.columns)
# Y_test=strat_test_set['MEDV'].copy()
# X_test_prepared=my_pipeline.transform(X_test)
# final_predictions=model.predict(X_test_prepared)
# final_mse=mean_squared_error(Y_test, final_predictions)
# final_rmse=np.sqrt(final_mse)
#final_rmse
#print(final_predictions,'\n',np.array(Y_test))

from joblib import dump, load
dump(model, 'finalmodel.joblib') 
dump(model_columns, 'model_columns.joblib')
lr = load('finalmodel.joblib')


