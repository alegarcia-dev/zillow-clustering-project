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
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

from _model import Model

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

################################################################################

def create_models(df):
    models = []
    features = [
        'square_feet',
        'non_average_zip_code',
        'tax_assessed_value'
    ]
    target = 'logerror'

    models.append(Model(LinearRegression(), df, features, target))
    models.append(Model(TweedieRegressor(), df, features, target))
    models.append(Model(make_pipeline(PolynomialFeatures(include_bias = False), LinearRegression()), df, features, target))
    models.append(Model(make_pipeline(PolynomialFeatures(include_bias = False, interaction_only = True), LinearRegression()), df, features, target))

    features = [
        'square_feet',
        'non_average_zip_code',
        'tax_assessed_value',
        'cluster_1',
        'cluster_2',
        'cluster_3'
    ]

    models.append(Model(LinearRegression(), df, features, target))
    models.append(Model(TweedieRegressor(), df, features, target))
    models.append(Model(make_pipeline(PolynomialFeatures(include_bias = False), LinearRegression()), df, features, target))
    models.append(Model(make_pipeline(PolynomialFeatures(include_bias = False, interaction_only = True), LinearRegression()), df, features, target))

    return models

################################################################################

def evaluate_models(baseline, models, train, validate, target):
    results = {
        'baseline' : {
            'RMSE_train' : mean_squared_error(train[target], baseline, squared = False),
            'RMSE_validate' : mean_squared_error(validate[target], pd.Series([baseline[0]] * validate.shape[0]), squared = False)
        }
    }

    labels = [
        'linear_regression_no_clusters',
        'tweedie_regressor_no_clusters',
        'polynomial_regression_no_clusters',
        'polynomial_regression_interactions_only_no_clusters',
        'linear_regression_with_clusters',
        'tweedie_regressor_with_clusters',
        'polynomial_regression_with_clusters',
        'polynomial_regression_interactions_only_with_clusters'
    ]

    for index, label in enumerate(labels):
        results[label] = {
            'RMSE_train' : mean_squared_error(train[target], models[index].make_predictions(train), squared = False),
            'RMSE_validate' : mean_squared_error(validate[target], models[index].make_predictions(validate), squared = False)
        }

    return pd.DataFrame(results).T

################################################################################

def evaluate_on_test(model, test, target):
    print(f'RMSE_test: {mean_squared_error(test[target], model.make_predictions(test), squared = False)}')