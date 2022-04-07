################################################################################
#
#
#
#       model.py
#
#       Description: description
#
#       Variables:
#
#           variables
#
#       Functions:
#
#           establish_baseline(target)
#
#
#
################################################################################

import pandas as pd

from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

################################################################################

def establish_baseline(target: pd.DataFrame) -> pd.Series:
    '''
        Determine whether to use the mean of the target or the median of the 
        target as the baseline model for a regression problem.
    
        Parameters
        ----------
        target: DataFrame
            The target variable for a regression problem.
    
        Returns
        -------
        Series: A pandas Series containing the best performer between the 
            median and mean of the target variable.
    '''

    baseline = pd.DataFrame({
        'median' : [target.median()] * target.size,
        'mean' : [target.mean()] * target.size
    })

    median_rmse = mean_squared_error(target, baseline["median"], squared = False)
    mean_rmse = mean_squared_error(target, baseline["mean"], squared = False)

    return baseline['median'] if median_rmse < mean_rmse else baseline['mean']