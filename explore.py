################################################################################
#
#
#
#       explore.py
#
#       Description: This file contains functions used in the zillow clustering
#           project final report for exploration.
#
#       Variables:
#
#           None
#
#       Functions:
#
#           None
#
#
#
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from preprocessing import remove_outliers

################################################################################

def plot_tax_value_and_logerror(df):
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 4))
    fig.suptitle('There is a wider range of logerror for lower valued properties.')

    sns.scatterplot(
        data = df,
        x = 'tax_assessed_value',
        y = 'logerror',
        alpha = 0.3,
        ax = ax[0]
    )
    ax[0].set_title('With Outliers')
    ax[0].set_xticks([0, 10_000_000, 20_000_000])
    ax[0].ticklabel_format(style = 'plain')

    sns.scatterplot(
        data = remove_outliers(df, 1.5, ['tax_assessed_value']),
        x = 'tax_assessed_value',
        y = 'logerror',
        alpha = 0.3,
        ax = ax[1]
    )
    ax[1].set_title('Without Outliers')
    ax[1].set_xticks([0, 250_000, 500_000, 750_000, 1_000_000])
    ax[1].ticklabel_format(style = 'plain')

    # plt.tight_layout()
    plt.show()

################################################################################

def run_stats_test_for_tax_value(df):
    mask = df.tax_assessed_value < 400_000
    _, p = stats.ttest_ind(df[mask].logerror, df[~mask].logerror, equal_var = False)

    if p < 0.05:
        print('Reject H0')
    else:
        print('Fail to reject H0')

################################################################################

def plot_zip_code_and_logerror(df):
    plt.figure(figsize = (14, 4))

    sns.scatterplot(
        data = remove_outliers(df, 1.5, ['zip_code']),
        x = 'zip_code',
        y = 'logerror',
        alpha = 0.3
    )

    plt.title('There are some zip codes with wider ranges of logerror.')
    plt.show()

################################################################################

def run_stats_test_for_zip_codes(df):
    alpha = 0.05
    zip_codes = []
    mean_logerror = df.logerror.mean()

    for zip_code in df.zip_code.unique():
        mask = df.zip_code == zip_code
        t, p = stats.ttest_1samp(df[mask].logerror, mean_logerror)
        
        if p < alpha:
            zip_codes.append(zip_code)

    print(f'{len(zip_codes)} / {len(df.zip_code.unique())} zip codes have mean log error significantly different than the overall mean log error.')

################################################################################

def plot_square_feet_and_logerror(df):
    sns.scatterplot(
        data = df,
        x = 'square_feet',
        y = 'logerror',
        alpha = 0.3
    )

    plt.title('There is a wider range of logerror for smaller properties.')
    plt.show()

################################################################################

def run_stats_test_for_square_feet(df):
    mean_logerror = df.logerror.mean()
    
    mask = df.square_feet < 2000
    t, p = stats.ttest_1samp(df[mask].logerror, mean_logerror)

    if p < 0.05:
        print('Reject H0')
    else:
        print('Fail to reject H0')

################################################################################

def plot_property_size_and_property_age(df):
    figure = sns.relplot(
        data = remove_outliers(df, 1.5, ['square_feet', 'lot_size']),
        x = 'square_feet',
        y = 'lot_size',
        col = 'yearbuilt_binned'
    )

    figure.fig.suptitle('Older properties tend to be smaller homes on smaller lots. Newer properties tend to be larger homes on smaller lots.', fontsize = 16)
    plt.tight_layout()
    plt.show()

################################################################################

def plot_clusters(df):
    sns.relplot(
        data = remove_outliers(df, 1.5, ['square_feet', 'lot_size']),
        x = 'square_feet',
        y = 'lot_size',
        col = 'yearbuilt_binned',
        hue = 'cluster'
    )

    plt.show()

################################################################################

def run_stats_test_for_clusters(df):
    _, p = stats.f_oneway(
        df[df.cluster == 0].logerror,
        df[df.cluster == 1].logerror,
        df[df.cluster == 2].logerror,
        df[df.cluster == 3].logerror
    )

    if p < 0.05:
        print('Reject H0')
    else:
        print('Fail to reject H0')