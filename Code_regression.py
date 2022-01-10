import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import math
import sys
import os

filename = "blogData_train.csv"
smp = "BlogPostData/blogData_test-2012.02.21.00_00.csv"

# filename = "blogData_feat.csv"
train_data = pd.read_csv(filename,header=None)

# filter_index = [9, 20, 5, 4, 10, 14, 19, 0, 51, 15, 34, 21, 11, 6, 1, 16, 3, 29, 35, 30, 40, 25, 44, 13, 8, 23, 50, 53, 18, 54, 46, 36, 31, 22, 26, 41, 56, 33, 38, 28, 48, 45, 43, 47, 58, 55, 60, 59, 52, 153, 57, 245, 66, 157, 231, 100]
filter_index2 = [9, 20, 5, 4, 10, 14, 19, 0, 51, 15, 34, 21, 11, 6, 1, 16, 3, 29, 35, 30, 40, 25, 44, 13, 8, 23, 50, 53]
# print([y for x in filter_index for y in filter_index2 if x==y ])
wrapper_index = [9, 5, 51, 15, 11, 1, 50, 54, 34, 21]

#train_data = train_data.iloc[np.random.permutation(len(train_data))]


test_data_list = []
# if not os.path.exists("Test_data.csv"):
for file in os.listdir('BlogPostData'):
    if 'test' in file:
        df = pd.read_csv('BlogPostData/'+file, header=None)
        test_data_list.append(df)
test_data = pd.concat(test_data_list, axis=0, ignore_index=True)
# else:
#     test_data = pd.read_csv("Test_data.csv", header=None)

    # filename = "blogData_test-2021.02.01.00_00.csv"

# train_data = pd.read_csv("pca_28.csv", header=None)
# test_data = pd.read_csv("pca_28_t.csv", header=None)
# test_data = pd.read_csv(smp, header = None)

train_output = train_data[len(train_data.columns)-1]
train_data2 = train_data.iloc[:,filter_index]
test_data2 = test_data.iloc[:,filter_index]        

del train_data[len(train_data.columns)-1]

print(len(train_data))
# test_data.to_csv("Test_data.csv", header=None, index=False)
#test_data = test_data.iloc[np.random.permutation(len(test_data))]
test_output = test_data[len(test_data.columns)-1]
del test_data[len(test_data.columns)-1]

reg = LinearRegression()
rf = RandomForestRegressor()
gradBoost = GradientBoostingRegressor()
ada = AdaBoostRegressor()



#n_estimators=500

regressors = [reg,rf,gradBoost,ada]
regressor_names = ["Linear Regression","Random Forests","Gradient Boosting","Adaboost"]

#regressors = ada
#regressor_names = "Adaboost"

for regressor,regressor_name in zip(regressors,regressor_names):
    
    regressor.fit(train_data,train_output)
    predicted_values = regressor.predict(test_data)
    predicted = np.clip(predicted_values, 0,5000)

    # counter = 0
    predicted = pd.DataFrame(data = predicted_values, index = None, columns = None)
    # top = pd.concat([test_output,predicted], axis=1, sort=False, ignore_index=True)
    # top = top.sort_values(0, ascending=False)
    # for i in range(10):
        # if math.ceil(top.iloc[i,0])== math.ceil(top.iloc[i,1]):
            # counter = counter+1

    print ("Mean Absolute Error for ",regressor_name," : ",metrics.mean_absolute_error(test_output,predicted_values))
    print ("STD : ",np.std(test_output-predicted_values))
    print ("Median Absolute Error for ",regressor_name, " : ",metrics.median_absolute_error(test_output,predicted_values))
    print ("Mean Squared Error for ",regressor_name, " : ",metrics.mean_squared_error(test_output,predicted_values))
    print ("Root Mean Squared Error for ",regressor_name, " : ",metrics.mean_squared_error(test_output,predicted_values, squared=False))

    print ("Max Error for ",regressor_name, " : ",metrics.max_error(test_output,predicted_values))
    print ("R2 score for ",regressor_name, " : ",metrics.r2_score(test_output,predicted_values))
    # print ("HIT@10 for ",regressor_name, " : ",counter)
    print("\n")
