import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# importing utility functions
import sys
sys.path.append("utils")

from plot_utils import *

def main():
    # read data
    coefs = pd.read_csv('logs/coefficients.csv')
    importances = pd.read_csv('logs/importances.csv')
    perf = pd.read_csv('logs/performances.csv')
    df_clean = pd.read_csv('data/data_clean.csv')
    df = pd.read_csv('data/data.csv')


    # data exploration
    plot_cor_clusters(df)
    plot_features(df)

    # define palette
    colors = ['lightblue', '#1f78b4', '#b2df8a']

    # plot performances
    metrics = ['MAE', 'RMSE', 'R2']
    plot_performance_subplots(perf, metrics, colors)

    # plot coefficients
    plot_coefficients(colors, coefs)

    # plot importances
    plot_importances(colors[0:2], importances)

if __name__=="__main__":
    main()