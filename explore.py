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

    sns.scatterplot(
        data = df,
        x = 'tax_assessed_value',
        y = 'logerror',
        alpha = 0.3,
        ax = ax[0]
    )
    ax[0].set_title('something')

    sns.scatterplot(
        data = remove_outliers(df, 1.5, ['tax_assessed_value']),
        x = 'tax_assessed_value',
        y = 'logerror',
        alpha = 0.3,
        ax = ax[1]
    )
    ax[1].set_title('something else')

################################################################################

def run_stats_test_for_tax_value(df):
    mask = df.tax_assessed_value < 250_000
    _, p = stats.ttest_ind(df[mask].logerror, df[~mask].logerror, equal_var = False)

    if p < 0.05:
        print('Reject H0')
    else:
        print('Fail to reject H0')