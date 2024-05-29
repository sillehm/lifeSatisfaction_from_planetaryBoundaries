import pandas as pd
import numpy as np
import warnings

from sklearn.linear_model import LassoCV, Lasso, RidgeCV, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# importing utility functions
import sys
sys.path.append("utils")

from model_utils import *


def main():
    # load data splits
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    # divide into target and features
    X_train, y_train = train.iloc[:,1:].values, train.iloc[:,0].values
    X_test, y_test = test.iloc[:,1:].values, test.iloc[:,0].values

    ### define *** for logging
    features = train.columns.tolist()[1:]
    performances = []
    param_log = []
    coef_df = pd.DataFrame()
    importances = []

    ### linear models without regularisation
    # define model configurations
    model_configs = [(range(0, 5), 'linear-social'),
                    (range(5, X_train.shape[1]), 'linear-bio'),
                    (range(X_train.shape[1]), 'linear-full')]

    # fit and evaluate linear models based on each configuration
    for col_range, model_name in model_configs:
        linear_fit_evaluate(X_train, X_test, 
                            y_train, y_test, 
                            col_range, 
                            model_name, 
                            performances)


    ### linear models with regularisation 
    # Lasso with 5 fold cross-validation
    regularisation_fit_evaluate('lasso',
                                X_train, X_test, 
                                y_train, y_test, 
                                'linear-lasso-cv', 
                                performances,
                                param_log)

    # Ridge with 5 fold cross-validation
    regularisation_fit_evaluate('ridge',
                                X_train, X_test, 
                                y_train, y_test, 
                                'linear-ridge-cv', 
                                performances,
                                param_log)



    # extract coefficients from the three models with all predictors
    model_names = ['linear-full', 'linear-lasso-cv', 'linear-ridge-cv' ]
    for name in model_names:
        coef_df = extract_coefs(coef_df, name, features)

    # saving to df
    coef_df.to_csv('logs/coefficients.csv',index = False)


    ### ensemble models
    # random forest fitted with cross validated grid search 
    param_grid = { 
    'n_estimators': [20, 50, 100, 300, 750],
    'max_depth' : [3, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'max_features': [0.3, 0.6, 0.9], 
    'ccp_alpha': [0.001, 0.01, 0.1, 0.5, 1]}

    ensemble_gridCV_fit_evaluate(RandomForestRegressor, 
                                param_grid, 
                                X_train, X_test, 
                                y_train, y_test, 
                                'RandomForest-cv', 
                                performances,
                                features,
                                importances,
                                param_log)


    # xgboost fitted with cross validated grid search 
    param_grid = { 
    'n_estimators': [10, 20, 100, 200, 500],
    'max_depth' : [2, 3, 5, 10],
    'objective': ['reg:squarederror'],
    'colsample_bytree': [0.3, 0.6, 0.9],
    'learning_rate': [2e-5, 2e-4, 2e-3, 2e-2, 2e-1]}

    ensemble_gridCV_fit_evaluate(XGBRegressor, 
                                param_grid, 
                                X_train, X_test, 
                                y_train, y_test, 
                                'xgboost-cv', 
                                performances,
                                features,
                                importances,
                                param_log)


    # save performance log as df
    performances_df = pd.DataFrame(performances)
    performances_df.to_csv('logs/performances.csv',index = False)

    # save feature importances as df
    importances_df = pd.concat(importances)
    importances_df.to_csv('logs/importances.csv',index = False)

    # save parameter log as df
    param_log_df= pd.DataFrame(param_log)
    param_log_df.to_csv('logs/parameter_log.csv',index = False)


if __name__=="__main__":
    main()