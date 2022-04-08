################################################################################
#
#
#
#       bivariate_analysis.py
#
#       Description: This file contains useful functions for performing bivariate
#           analysis, which can be used in the exploration phase of the data 
#           science pipeline.
#
#       Variables:
#
#           None
#
#       Functions:
#
#           plot_continuous_and_continuous(df, x, y, title = '')
#           plot_categorical_and_continuous(df, categorical_cols, continuous_cols)
#
#
################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################

def plot_continuous_and_continuous(df: pd.DataFrame, x: str, y: str, title: str = '') -> None:
    '''
        Create a line plot and scatter plot with regression line for two 
        continuous variables.
    
        Parameters
        ----------
        df: DataFrame
            The dataframe from which to use the features provided to create 
            the line and scatter plots.
    
        x: str
            The column in the dataframe for which we would like to determine 
            the relationship with the target.

        y: str
            The target variable in the dataframe for which we are relating to 
            the x variable.

        title: str
            An optional title to provide for the figure.
    '''

    figure_height = 3
    figure_width = 14

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (figure_width, figure_height))
    mean = df[y].mean()

    fig.suptitle(title)

    sns.lineplot(data = df, x = x, y = y, ax = ax[0])
    ax[0].axhline(mean, ls='--', color='grey')

    sns.scatterplot(data = df, x = x, y = y, ax = ax[1], alpha = 0.3, color = 'blue')
    ax[1].axhline(mean, ls='--', color='grey')

    plt.show()

################################################################################

def plot_categorical_and_continuous(
    df: pd.core.frame.DataFrame,
    categorical_cols: list[str],
    continuous_cols: list[str]
) -> None:
    '''
        Plot a boxplot, barplot, and histplot for each combination of continous 
        and categorical column in the dataframe provided.
    
        Parameters
        ----------
        df: DataFrame
            A pandas dataframe containing the data to be plotted.

        categorical_cols: list[str]
            A list of the categorical columns to plot.

        continuous_cols: list[str]
            A list of the continuous columns to plot.
    '''

    for con in continuous_cols:
        for cat in categorical_cols:
            fig = plt.figure(figsize = (14, 4))
            fig.suptitle(f'{con} v. {cat}')

            plt.subplot(1, 3, 1)
            sns.boxplot(data = df, x = cat, y = con)
            plt.axhline(df[con].mean())

            plt.subplot(1, 3, 2)
            sns.barplot(data = df, x = cat, y = con)
            plt.axhline(df[con].mean())

            plt.subplot(1, 3, 3)
            sns.histplot(data = df, x = con, bins = 10, hue = cat)
            plt.show()