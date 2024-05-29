import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.linear_model import LassoCV, Lasso, RidgeCV, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


performances = []
features = []
importances = []
param_log = []


def evaluate(model, X, y, nsplit, model_name, performances):
    ''' Evaluates the performance of a model 
    Args:
        model (sklearn.Estimator): fitted sklearn estimator
        X (np.array): predictors
        y (np.array): true outcome
        nsplit (str): name of the split
        model_name (str): an identifier for the model
        performances (list): empty list to store performance results
    '''
    #global performances
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    performance = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    performances.append({'model': model_name,
                         'split': nsplit,
                         'RMSE': performance.round(4),
                         'R2': r2.round(4),
                         'MAE': mae.round(4)})

def linear_fit_evaluate(X_train, X_test, y_train, y_test, cols, model_name, performances):
    """ Function for fitting a linear regression model on the specified columns of the features and evaluates its performance

    Args:
        X_train (np.array): training set features
        X_test (np.array): test set features
        y_train (np.array): target variable of the training set
        y_test (np.array): target variable of the test set
        cols (range): range of feature column indices to be used
        model_name (str): name of the model
        performances (list): list to store performance results
    """
    # fit linear regression model
    reg = LinearRegression().fit(X_train[:, cols], y_train)

    # save the model
    pkl.dump(reg, file=open(f'models/{model_name}.pkl', 'wb'))

    # evaluate
    for x, y, nsplit in zip([X_train, X_test], 
                            [y_train, y_test], 
                            ['train', 'test']):
        evaluate(model=reg, 
                X=x[:, cols], y=y, 
                nsplit=nsplit, 
                model_name=model_name,
                performances=performances)

def regularisation_fit_evaluate(regressor, X_train, X_test, y_train, y_test, model_name, performances, param_log):
    
    ''' Function for fitting a regularised regression model (lasso or ridge) with cross-validation and evaluate its performance

    Args:
        regressor (str): type of regularized regression model - 'lasso' or 'ridge'.
        X_train (np.array): training set features
        X_test (np.array): test set features
        y_train (np.array): target variable of the training set
        y_test (np.array): target variable of the test set
        model_name (str): name of the model
        performances (list): list to store performance results
        param_log (list): list to store parameter logs
    '''
    
    if regressor == 'lasso':
        # Lasso with 5 fold cross-validation, for 100 alphas
        modelCV = LassoCV(cv=5, random_state=99, max_iter=10000)

    else: 
        # alpha values: 100 values from 0 to 5 evenly spaced on a log scale
        alphas = np.logspace(0, 5, 100)

        # initiate the 5-fold cross validation over alphas - scoring on neg_mse 
        modelCV = RidgeCV(alphas=alphas, cv=5)

    
    modelCV.fit(X_train, y_train)
    # extract best alpha
    alphaCV = modelCV.alpha_

     # fit model with the best alpha from cross-validation
    if regressor == 'lasso':
        model = Lasso(alpha=alphaCV)
    else:
        model = Ridge(alpha=alphaCV)
    model.fit(X_train, y_train)

    # save the model
    pkl.dump(model, file=open(f'models/{model_name}.pkl', 'wb'))

    # Evaluate on train, validation, and test sets
    for x,y,nsplit in zip([X_train, X_test],
                    [y_train, y_test],
                    ['train', 'test']):
        evaluate(model=model, 
                X=x, y=y, 
                nsplit=nsplit, 
                model_name=model_name, 
                performances=performances)
    
    # log best parameter
    param_log.append({'model': model_name, 'best_params': {'alpha': alphaCV}})


def extract_coefs(df, model_name, features):
    """ Function for extracting coefficients from a saved model and adding to dataframe

    Args:
        df (DataFrame): dataframe to store coefficients
        model_name (str): name of the saved model file
        features (list): list of feature names
    
    Returns:
        DataFrame: dataFrame with model coefficients
    """
    
    # Load the saved model
    with open(f'models/{model_name}.pkl', 'rb') as file:
        model = pkl.load(file)
    
    # Extract model type
    model_type = str(model).split('(')[0]  # Get the first part of the string

    # Extract coefficients
    coefs = model.coef_.round(4)
    
    # Add coefficients to df
    for feature, coef in zip(features, coefs):
        df = df.append({'feature': feature, 'coefficient': coef, 'model': model_type}, ignore_index=True)

    return df


def ensemble_gridCV_fit_evaluate(regressor, param_grid, X_train, X_test, y_train, y_test, model_name, performances, features, importances, param_log):
    
    ''' Functionfor fitting an ensemble model using GridSearchCV and evaluating its performance on train, validation, and test sets
        Args:
            regressor (sklearn.Estimator): regressor class (RandomForestRegressor or XGBRegressor)
            param_grid (dict): dictionary with parameter names (`str`) as keys and lists of parameter settings to try as values
            X_train (np.array): training set features
            X_test (np.array): test set features
            y_train (np.array): target variable of the training set
            y_test (np.array): target variable of the test set
            model_name (str): name of the model
            performances (list): list to store performance results
            features (list): list of features
            importances (list): list for storing feature importances
            param_log (list): lidt for storing model parameters
    '''
    
    # instantiate estimator
    reg = regressor(random_state=99) 

    # cross validation
    model = GridSearchCV(estimator=reg, 
                            param_grid=param_grid,
                            scoring='neg_mean_squared_error',
                            cv=5)
    model.fit(X_train, y_train)

    # save the model
    pkl.dump(reg, file=open(f'models/{model_name}.pkl', 'wb'))

    # evaluate
    for x, y, nsplit in zip([X_train, X_test], 
                            [y_train, y_test], 
                            ['train', 'test']):
        evaluate(model=model.best_estimator_, 
                X=x, y=y, 
                nsplit=nsplit, 
                model_name=model_name,
                performances=performances)
    
   
    # Extract feature importances
    feature_importances = model.best_estimator_.feature_importances_
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importances
    })
    importance_df['model'] = str(model_name).split('-')[0]
    importances.append(importance_df)

    # log best parameters
    best_params = model.best_params_
    param_log.append({'model': model_name, 'best_params': best_params})


