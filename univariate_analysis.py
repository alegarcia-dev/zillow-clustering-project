################################################################################
#
#
#
#       univariate_analysis.py
#
#       Description: This file contains useful functions for performing univariate
#           analysis, which can be used pre-split in the preparation phase of the
#           data science pipeline.
#
#       Variables:
#
#           None
#
#       Functions:
#
#           get_hist(df, columns)
#           get_box(df, columns)
#           _create_sub_plots(num_columns)
#           plot_single_variable(df, feature, title = '', histplot_bins = 50)
#
#
################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

################################################################################

def get_hist(df: pd.core.frame.DataFrame, columns: list[str]) -> None:
    '''
    Gets histographs of acquired continuous variables.
    
    Parameters
    ----------
    df: DataFrame
        A pandas dataframe from which we will visualize the univariate 
        distributions of features.

    columns: list[str]
        A list of the columns we would like to visualize.
    '''

    fig, ax = _create_sub_plots(len(columns))
    num_columns = 3

    for index, column in enumerate(columns):
        sns.histplot(
            data = df[column],
            bins = 5,
            ax = ax[index // num_columns][index % num_columns]
        )

        # Rotate the tick labels in case there are some lengthy tick labels.
        ax[index // num_columns][index % num_columns].tick_params(labelrotation = 30)

        # Title with column name.
        plt.title(column)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.grid(False)
        plt.tight_layout()

    plt.show()

################################################################################

def get_box(df: pd.core.frame.DataFrame, columns: list[str]) -> None:
    '''
    Gets boxplots of acquired continuous variables.
    
    Parameters
    ----------
    df: DataFrame
        A pandas dataframe from which we will visualize the univariate 
        distributions of features.

    columns: list[str]
        A list of the columns we would like to visualize.
    '''

    fig, ax = _create_sub_plots(len(columns))
    num_columns = 3

    for index, column in enumerate(columns):
        sns.boxplot(
            data = df,
            x = column,
            ax = ax[index // num_columns][index % num_columns]
        )

        # Rotate the tick labels in case there are some lengthy tick labels.
        ax[index // num_columns][index % num_columns].tick_params(labelrotation = 30)

        # Title with column name.
        plt.title(column)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.grid(False)
        plt.tight_layout()

    plt.show()

################################################################################

def _create_sub_plots(num_features: int):
    '''
        Create figure with subplots for the number of columns provided 
        where each row of subplots has 3 columns and there are enough 
        rows to fit all columns into a plot.
    
        Parameters
        ----------
        num_columns: int
            The number of columns to fit in the figure.
    '''

    # We want the height to be a multiple of 3 and it should provide
    # enough space for each of our variables.
    figure_width = 14
    figure_height = ((num_features // 3) + 1) * 3

    # Similar to the height we want the number of rows to account for
    # each row having 3 plots.
    num_rows = num_features // 3 + 1
    num_columns = 3
    
    fig, ax = plt.subplots(nrows = num_rows, ncols = num_columns, figsize = (figure_width, figure_height))
    return fig, ax

################################################################################

def plot_single_variable(df: pd.DataFrame, feature: str, title: str = '', histplot_bins: int = 50):
    '''
        description
    
        Parameters
        ----------
        parameter: parameter_type
            parameter_description
    
        Returns
        -------
        return_type: return_description
    '''

    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 4))
    fig.suptitle(title)

    sns.histplot(df[feature], bins = histplot_bins, ax = ax[0])
    sns.boxplot(data = df, x = feature, ax = ax[1])

    plt.show()