import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data_sheets(path):
    ''' Function for loading excel file, extracting data from sheets and merging to one df
    Args:
        path (str): path to excel file
    
    Returns:
        pd.DataFrame: the merged DataFrame containing data from the second and third sheets of the excel file,
                      joined on the 'Country' column
    '''
    # load the Excel file
    xlsx_file = pd.ExcelFile(path)
    # sheet 2 to df
    biophysical = xlsx_file.parse(xlsx_file.sheet_names[1])
    # sheet 3 to df
    social = xlsx_file.parse(xlsx_file.sheet_names[2]) 

    # naming the first column country
    biophysical.rename(columns={'Unnamed: 0': 'Country'}, inplace=True)
    social.rename(columns={'Unnamed: 0': 'Country'}, inplace=True)

    # Joining on country column
    df = pd.merge(social, biophysical, on='Country', how='inner')

    return df



def remove_na(df, target_column):
    ''' Removes missing values iteratively from the dataframe
    
    Args:
        df (pd.DataFrame): input dataframe
        target_column (str): name of the target column
    
    Returns:
        pd.DataFrame: the dataframe with missing values removed
    '''

    # remove rows with missing values in the target variable
    df = df.dropna(subset=[target_column])

    # remove rows with 3 or fewer missing values ['CO2 Emissions','Social Support', 'Ecological Footprint'] 
    df = df[df.isna().sum(axis=1) <= 3]

    # remove columns with only one missing value ['Nitrogen', 'Phosphorus', 'Material Footprint']
    df.dropna(subset=['Nitrogen', 'Phosphorus', 'Material Footprint'], inplace=True)

    # remove columns with any remaining missing values ['Nutrition', 'Sanitation','Income', 'Education', 'Equality','Blue Water']
    df = df.drop(columns = df.columns[df.isna().sum() > 0])

    return df


def prepare_data(df, feature_ind, target, test_size):
    ''' Function preparing and scaling data for modeling.
    The data is divided into target and features, split into train and test sets and 
    the features are scaled using Min-Max scaling

    Args:
        df (pd.DataFrame): preprocessed dataframe
        feature_ind (list): start and end of feature columns indices
        target (int): index of the target column
        test_size (int): proportion of the dataset to include in the test and validation sets

    Returns
        tuple: scaled training and test sets for features and target 
    '''

    # divide into target and features
    X = df.iloc[:, feature_ind[0]:feature_ind[1]].values
    y = df.iloc[:, target].values 

    # divide into train, test and validation sets
        # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=42)

    # scaling features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test



def save_data_splits(X_train, X_test, y_train, y_test, column_names, output_dir='data'):
    ''' Function saving data splits to csv
    Args:
        X_train (np.array): training set features
        X_test (np.array): test set features
        y_train (np.array): target variable of the training set
        y_test (np.array): target variable of the test set
        column_names (pd.Index): target and feature column names
        output_dir (str, optional): directory to save the CSV files with 'data' as default
    '''

    # iterate over train, validation, and test sets with their labels (name)
    for x, y, name in zip([X_train, X_test], [y_train, y_test], ['train', 'test']):
        # bombine features and target variable horizontally to create a df
        data = pd.DataFrame(np.hstack([x, y.reshape(-1, 1)]))  # reshape to ensure the same amount of rows
        
        # assign column names from target and features to the df
        data.columns = column_names.columns
        
        # write df to CSV with a name based on the set (train, val, test)
        data.to_csv(f'{output_dir}/{name}.csv', index=False)