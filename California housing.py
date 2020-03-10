from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

price = datasets.fetch_california_housing()

features = price.feature_names
for i in price.data:
    i[0] = i[0] * 10000

price.target = price.target * 100000

xtrain,xtest,ytrain,ytest = train_test_split(price.data,price.target,test_size=0.3,random_state=0)

regr = linear_model.LinearRegression()
model = regr.fit(xtrain,ytrain)
# f = pd.concat([pd.DataFrame(price.data),pd.DataFrame(price.target)],axis=1)
# print(f)
print(model)
print(regr.intercept_)
print(features)
print(regr.coef_)
y_predLR = regr.predict(xtest)
sum_erroLR = np.sqrt(mean_squared_error(ytest,y_predLR))
print("The Root Mean Squared Error of Linear Regression Model is: ",sum_erroLR)
scoresLR = cross_val_score(regr,price.data,price.target,scoring = "neg_mean_squared_error",cv = 8)
scoresLR = np.sqrt(-scoresLR)
print("The result of cross validation: ",scoresLR.mean())

print("\n")
forest = RandomForestRegressor()
model0 = forest.fit(xtrain,ytrain)
y_predRF = forest.predict(xtest)
sum_erroRF = np.sqrt(mean_squared_error(ytest,y_predRF))
scoresRF = cross_val_score(forest,price.data,price.target,scoring = "neg_mean_squared_error",cv = 8)
scoresRF = np.sqrt(-scoresRF)
print(model0)
print("The Root Mean Squared Error of Random Forest Model is: ",sum_erroRF)
print("The result of cross validation: ",scoresRF.mean())













