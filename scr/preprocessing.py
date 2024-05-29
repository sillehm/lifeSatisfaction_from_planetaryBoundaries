import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# importing utility functions
import sys
sys.path.append("utils")

from data_utils import load_data_sheets, remove_na, prepare_data, save_data_splits

def main():
    # load data
    in_path = 'data/GoodLifeWithinPB_SupplementaryData.xlsx'
    out_path = 'data/data.csv'
    data = load_data_sheets(in_path)
    data.to_csv(out_path, index=False)

    # drop missing values in target column and in columns with less than 3 missing values
    # drop columns with many mising values 
    df_clean = remove_na(data, 'Life Satisfaction')

    # Save data to csv
    path = 'data/data_clean.csv'
    df_clean.to_csv(path, index=False)

    # prepare data for modeling
    X_train, X_test, y_train, y_test = prepare_data(df_clean, 
                                                    feature_ind=[2,df_clean.shape[1]], 
                                                    target=1, 
                                                    test_size=0.2)

    # save data splits
    column_names = df_clean.drop(columns=['Country'])
    save_data_splits(X_train, X_test, y_train, y_test, column_names)            

if __name__=="__main__":
    main()
