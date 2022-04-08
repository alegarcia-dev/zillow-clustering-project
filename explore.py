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