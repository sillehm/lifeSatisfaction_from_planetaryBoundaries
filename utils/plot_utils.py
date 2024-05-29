import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Functions for data exploration

def plot_features(df, output_dir='plots/'):
    # defining grid dimensions and figure size
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(12, 8)) 
    # extracting feature data (columns 2 thorugh 18)
    features = df.iloc[:,2:] 
    # defining target data
    target = df['Life Satisfaction']

    # defining colour 
    color = '#1f78b4'

    # iterating over features and plotting them on subplots
    for i, c in enumerate(features.columns):
        sns.scatterplot(data=df, x=c, y=target, ax=ax[i // 4][i % 4], color=color)
        ax[i // 6][i % 6].set_xlabel(c)
        ax[i // 6][i % 6].set_ylabel('Life Satisfaction')

    # removing the last subplot
    ax.flat[-1].set_visible(False)

    # adjusting layout
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{output_dir}target_feature_associations.png')

def plot_cor_clusters(df, output_dir='plots/'):
    corr = df.corr()
    sns.clustermap(corr, cmap='viridis', annot=True, fmt=".2f", annot_kws={"fontsize": 8})
    plt.show()
    plt.savefig(f'{output_dir}cor_clustermap.png')

def plot_cor_heat(df, output_dir = 'plots/'):
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='viridis', square=True, cbar_kws={"shrink": 0.8}, annot_kws={"fontsize": 8})
    plt.title('Correlation Heatmap')
    plt.show()
    plt.savefig(f'{output_dir}cor_heatmap.png')



def plot_performance_subplots(performances, metrics, colors, output_dir='plots/'):
    ''' Function for plotting the performance of models based on three metrics (RMSE, R², and MAE) in subplots
    Args:
        performances (pd.DataFrame): dataframe with stored model performances
        metrics (list): list of metrics to be plotted
        colors (list): list of colors 
        output_dir (str, optional): directory to save the figure
    '''
    
    # Clear the previous plot
    plt.clf()

    # define order of split categories
    split_order = ['train', 'test']

    sns.set_style('whitegrid')
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    for ax, metric in zip(axes, metrics):
        sns.scatterplot(data=performances.sort_values(by=metric, ascending=False), 
                        y='model', 
                        x=metric, 
                        marker='s', 
                        hue='split', 
                        palette=colors,
                        hue_order=split_order,
                        s = 100,
                        ax=ax)
        ax.set_title(metric, fontsize = 20)
        ax.legend(loc='lower right', fontsize = 16)
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_ylabel('')  # Remove y-axis label
        ax.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.show()
    fig.savefig(f'{output_dir}performance_plot_metrics.png')



######
# performance
def plot_performance(performances, metric, colors, output_dir='plots/'):
    ''' Function for plotting the performance of models based on a specified metric
    Args:
        performances (pd.DataFrame): dataframe with stored model performances
        metric (str): performance metric to be presented ('rmse' or 'r2')
        colors (list): list of colors 
        output_dir (str, optional): directory to save the figure
    '''
    
    # Clear the previous plot
    plt.clf()


    # define order of split categories
    split_order = ['train', 'test']

    sns.set_style('whitegrid')
    sns.scatterplot(data=performances.sort_values(by=metric, ascending=False), 
                    y='model', 
                    x=metric, 
                    marker='s', 
                    hue='split', 
                    palette=colors,
                    hue_order=split_order)

    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(f'{output_dir}performance_plot_{metric}.png')


def plot_coefficients(colors, coefs, output_dir='plots/'):
    ''' Function for visualising coefficients of features in a barplot
    Args:
        colors (list): list of colors 
        coefs (pd.DataFrame): dataframe with stored model coefficients
        output_dir (str, optional): directory to save the figure
    '''
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))#, sharey=True)

    # Iterate over unique values in the "Model" column
    for i, model_name in enumerate(coefs['model'].unique()):
        # Filter DataFrame for the current model
        model_df = coefs[coefs['model'] == model_name]
        
        # Create barplot for the current model
        sns.barplot(x='feature', y='coefficient', data=model_df, ax=axes[i], color=colors[i])
        
        # Set plot title
        axes[i].set_title(model_name)
        
        # Rotate x-axis labels for better readability
        axes[i].tick_params(axis='x', rotation=45)

        axes[i].set_xticklabels(axes[i].get_xticklabels(), ha='right')  # Adjust alignment

    for ax in axes:
        ax.set_ylim(-0.2, 0.7)
        
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()

    # save figure
    plt.savefig(f'{output_dir}coef_barplot.png')





# importances
def plot_importances(colors, importances, output_dir='plots/'):
    ''' Function for visualising feature importances in a barplot
    Args:
        colors (list): list of colors 
        importances (pd.DataFrame): dataframe with stored feature importances
        output_dir (str, optional): directory to save the figure
    '''

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))#, sharey=True)

    # Iterate over unique values in the "Model" column
    for i, model_name in enumerate(importances['model'].unique()):
        # Filter DataFrame for the current model
        model_df = importances[importances['model'] == model_name]
        
        # Create barplot for the current model
        sns.barplot(x='feature', y='importance', data=model_df, ax=axes[i], color=colors[i])
        
        # Set plot title
        axes[i].set_title(model_name)
        
        # Rotate x-axis labels for better readability
        axes[i].tick_params(axis='x', rotation=45)

        axes[i].set_xticklabels(axes[i].get_xticklabels(), ha='right')  # Adjust alignment

    for ax in axes:
        ax.set_ylim(0, 0.7)

    # Adjust layout
    plt.tight_layout()

    # save figure
    plt.savefig(f'{output_dir}importance_barplot.png')

    # Show plot
    plt.show()