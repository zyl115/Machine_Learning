#import libraries and functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

#import data
data_white = pd.read_csv('winequality-white.csv', sep = ';')
data_red = pd.read_csv('winequality-red.csv', sep = ';')
data_comb = pd.concat ([data_white, data_red])
y = data_comb.quality
X = data_comb.drop ('quality', axis = 1)

#split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size = 0.2,
                                                    random_state = 29 )


#scale data
scaler = StandardScaler()
scaler.fit (X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform (X_test)

#linear regression
lin_regr = linear_model.LinearRegression()
lin_regr.fit(X_train,y_train)
y_pred_train = np.around (lin_regr.predict(X_train))
y_pred_test = np.around( lin_regr.predict (X_test))
print("Linear Regression Mean squared error (Train): %.5f"
      % mean_squared_error(y_train, y_pred_train))
print("Linear Regression Mean squared error (Test): %.5f"
      % mean_squared_error(y_test, y_pred_test))

#store coefficients and intercept term
w1 = lin_regr.coef_
bias = lin_regr.intercept_


#polynomial regression of degree 4
poly = PolynomialFeatures (degree = 4)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform (X_test)
lin_regr.fit(X_train_poly,y_train)
ypoly_pred_train = lin_regr.predict(X_train_poly)
ypoly_pred_test = lin_regr.predict (X_test_poly)
print("Polynomial Regression Mean squared error (Train): %.5f"
      % mean_squared_error(y_train, ypoly_pred_train))
print("Polynomial Regression Mean squared error (Test): %.5f"
      % mean_squared_error(y_test, ypoly_pred_test))

#train and test error for different alpha (Ridge regression)
alpha_vals = [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,50,100,500,1000]
trainerror = []
testerror = []
for k in alpha_vals:
    reg = linear_model.Ridge (alpha = k, normalize = True)
    reg.fit (X_train_poly,y_train)
    ypoly_pred_train = reg.predict(X_train_poly)
    ypoly_pred_test = reg.predict (X_test_poly)
    trainerror.append(mean_squared_error(y_train, ypoly_pred_train))
    testerror.append(mean_squared_error(y_test, ypoly_pred_test))

#plot training and test error against alpha
plt.title("Training and Test Error of Ridge Regression against alpha")
plt.xlabel("alpha")
plt.ylabel("Error")
plt.ylim(0.0, 1.1)
navy_patch = mpatches.Patch(color='navy', label='Test Error')
darkorange_patch = mpatches.Patch(color='darkorange', label='Training Error')
plt.legend(handles=[navy_patch,darkorange_patch])
lw = 2
plt.semilogx(alpha_vals, trainerror, label="Training Error",
             color="darkorange", lw=lw)

plt.semilogx(alpha_vals, testerror, label="Test Error",
             color="navy", lw=lw)
plt.show()


#ridge regression with cross validation
reg = linear_model.RidgeCV (alphas = alpha_vals, normalize = True, cv=None,  store_cv_values = True)
reg.fit (X_train_poly,y_train)
ypoly_pred_train = reg.predict(X_train_poly)
ypoly_pred_test = reg.predict (X_test_poly)
#print('Coefficients: \n', reg.coef_)
print("Ridge Regression Mean squared error (Train): %.5f"
      % mean_squared_error(y_train, ypoly_pred_train))
print("Ridge Regression Mean squared error (Test): %.5f"
      % mean_squared_error(y_test, ypoly_pred_test))
# Explained variance score: 1 is perfect prediction
print ('Ridge Regression Alpha value: %.2f' % reg.alpha_)

#calculate cross validation score
cv_values = reg.cv_values_
cv_scores = cv_values.mean(axis=0)

#plot cross validation curve for ridge regression
plt.title("Validation Curve with Ridge Regression")
plt.xlabel("alpha")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(alpha_vals, testerror, label="Test Error",
             color="navy", lw=lw)
plt.semilogx(alpha_vals, cv_scores, label="Cross-validation score",
             color="red", lw=lw)
navy_patch = mpatches.Patch(color='navy', label='Test Error')
red_patch = mpatches.Patch(color='red', label='Cross-validation score')
plt.legend(handles=[navy_patch,red_patch])
plt.show()


#MLP 
mlpreg = MLPRegressor(alpha=0.3,hidden_layer_sizes=(8,), random_state=6 , max_iter = 100000)
mlpreg.fit(X_train,y_train)
ymlp_pred_train = mlpreg.predict(X_train)
ymlp_pred_test = mlpreg.predict (X_test)
print("MLP Mean squared error (Train): %.5f"
      % mean_squared_error(y_train, ymlp_pred_train))
print("MLP Mean squared error (Test): %.5f"
      % mean_squared_error(y_test, ymlp_pred_test))


#test mlp with different number of hidden layers
mlp_param = [(10,),(10,10),(10,10,10),(10,10,10,10),(10,10,10,10,10)]
crossvalscore = []
numoflayers = [1,2,3,4,5]

for k in mlp_param:
    score = -cross_val_score(
        MLPRegressor(hidden_layer_sizes=k, random_state=6 , max_iter = 100000), X_train, y_train,
        cv=10, scoring="neg_mean_squared_error", n_jobs=1).mean()
    crossvalscore.append(score)

#plot validation curve of mlp different num of layers
plt.title("Validation Curve with MLP (number of hidden layers)")
plt.xlabel("number of layers")
plt.ylabel("Score")
lw = 2
plt.plot(numoflayers, crossvalscore, label="Cross-validation score",
             color="red", lw=lw)
plt.show()

#test mlp with different number of nodes
mlp_param = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150]
crossvalscore = []


for k in mlp_param:
    score = -cross_val_score(
        MLPRegressor(hidden_layer_sizes=k, random_state=6 , max_iter = 100000), X_train, y_train,
        cv=10, scoring="neg_mean_squared_error", n_jobs=1).mean()
    crossvalscore.append(score)

#plot validation curve of mlp different num of layers
plt.title("Validation Curve with MLP (number of nodes)")
plt.xlabel("number of nodes")
plt.ylabel("Score")
lw = 2
plt.plot(mlp_param, crossvalscore, label="Cross-validation score",
             color="red", lw=lw)
plt.show()

#MLP with best parameter
mlpreg = MLPRegressor(hidden_layer_sizes=(90,), random_state=6 , max_iter = 100000)
mlpreg.fit(X_train,y_train)
ymlp_pred_train = mlpreg.predict(X_train)
ymlp_pred_test = mlpreg.predict (X_test)
print("MLP Best Mean squared error (Train): %.5f"
      % mean_squared_error(y_train, ymlp_pred_train))
print("MLP Best Mean squared error (Test): %.5f"
      % mean_squared_error(y_test, ymlp_pred_test))
