# Boosting Algorithms and LightGBM Project
### By Larry Feng

## My Boosting Algorithm

For my boosting algorithms, I chose to test a few combinations for fun and out of curiosity. All of the models are based on the `boosted_lwr()` method written in class, but using different kernels and regressors. The first model uses the XGBoost regressor, the second uses Random Forest Regressor, and the third uses Decision Tree Regresor (DTR). 

```
model_boosting1 = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=150,reg_lambda=0.1,alpha=10,gamma=18,max_depth=3)
model_boosting2 = RandomForestRegressor(n_estimators=150, min_samples_split=3, max_depth=3,random_state=410)
model_boosting3 = DTR(max_depth=4, min_samples_split=3, min_samples_leaf=2,random_state=410)
```

After declaring the models, we will be using them for predicting and then calculating the Mean Squared Errors for each. Below is the code for calculating the cross-validated values for the three models:

```
mse_blwr_xgb = []
mse_blwr_rf = []
mse_blwr_dtr = []

for i in range(2):
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = ss.fit_transform(X[idxtrain])
    ytrain = y[idxtrain]
    xtest = ss.transform(X[idxtest])
    ytest = y[idxtest]
    xtest = X[idxtest]
    
    yhat_blwr_xgb = boosted_lwr(xtrain,ytrain,xtest,Quartic,1,True,model_boosting1,3)
    yhat_blwr_rf = boosted_lwr(xtrain,ytrain,xtest,Tricubic,1,True,model_boosting2,3)
    yhat_blwr_dtr = boosted_lwr(xtrain,ytrain,xtest,Epanechnikov,1,True,model_boosting2,3)

    mse_blwr_xgb.append(mse(ytest,yhat_blwr_xgb))
    mse_blwr_rf.append(mse(ytest,yhat_blwr_rf))
    mse_blwr_dtr.append(mse(ytest,yhat_blwr_dtr))

print('The Cross-validated Mean Squared Error for Boosted LWR with XGboost is : '+str(np.mean(mse_blwr_xgb)))
print('The Cross-validated Mean Sqaured Error for Boosted LWR with Random Forest is : '+str(np.mean(mse_blwr_rf)))
print('The Cross-validated Mean Squared Error for Boosted LWR with Decision Trees is : '+str(np.mean(mse_blwr_dtr)))
```
After running for 24 minutes, these are the results I have achieved- the MSE for each model is very high. Below is the output:
```
The Cross-validated Mean Squared Error for Boosted LWR with XGboost is : 1177.7802237152455
The Cross-validated Mean Sqaured Error for Boosted LWR with Random Forest is : 839.0729183542422
The Cross-validated Mean Squared Error for Boosted LWR with Decision Trees is : 916.6757999106142
```
Since the lowest cross-validated mean squared error belongs to the Random Forest model, I conclude that the Random Forest Regressor Boosted model is the best among the models I have tried. Although I would like to try some more models, the long run-time has deterred me from doing so.

## LightGBM

### Theory

LightGBM (Light Gradient Boosting Machine) is a gradient boosting framework that is based on decision trees; this increases efficiency of the model and reduces the memory used. It consists of two techniques: Gradient-based One Side Sampling(GOSS) and Exclusive Feature Bundling (EFB). LightGBM has many of XGBoost's advantages, such as sparse optimization, parallel training, multiple loss functions, bagging, early stopping, and regularization. But a difference between the two is the construction of their trees- LightGBM grows trees leaf wise, so it chooses the leaf that it thinks will yield the largest decrease in loss. 

GOSS: This method keeps in mind the fact that there isn't a native weight for data instance in Gradient Boosted Decision Trees. Since data instances have different gradients that play different roles in the computation of information gain, the instances with larger gradients will have more information gain. To keep a stable accuracy of information, this method keeps instances with large gradients and randomly drops the instances with small gradients.

EFB: Exclusive Feature Bundling is a near-lossless method that works to reduce the number of effective features. In a sparse feature space, many features are nearly exclusive, meaning that the features rarely take non-zero values simultaneously. One example of an exclusive feature is on-hot encoded features. EFB bundles these features, and it reduces dimensionality to improve efficiency while maintaining a high accuracy. The act of bundling exclusive features into a single feature is why it's called exclusive feature bundling. 

### Application

Here, I will apply the LightGBM algorithm to the dataset to get a cross-validated MSE.
```
mse_LGBM = []

for i in range(2):
  kf = KFold(n_splits=10,shuffle=True,random_state=i)
  # this is the Cross-Validation Loop
  for idxtrain, idxtest in kf.split(X):
    xtrain = ss.fit_transform(X[idxtrain])
    ytrain = y[idxtrain]
    xtest = ss.transform(X[idxtest])
    ytest = y[idxtest]
    xtest = X[idxtest]

    model = lgb.LGBMRegressor()
    model.fit(xtrain,ytrain)
    yhat = model.predict(xtest)

    mse_LGBM.append(mse(ytest,yhat))

print('The Cross-validated Mean Squared Error for LightGBM is : '+str(np.mean(mse_LGBM)))
```
Overall, the process is relatively the same as using the other algorithms, but I had to remember to fit the model since the previous KFold code blocks use the `booster()` method which already has a `fit()` function inside. Regardless, the output for our LightGBM regressor is:

```
The Cross-validated Mean Squared Error for LightGBM is : 310.13562109367183
```
Although the Mean squared error is still relatively high, it is our lowest result by far. The lowest result of our boosting algorithms is around 800, so this result is really good to see. 
