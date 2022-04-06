################################################################################
#
#
#
#       prepare.py
#
#       Description: This file contains the functions used for preparing the zillow
#           dataset for the clustering project.
#
#       Variables:
#
#           None
#
#       Functions:
#
#           summarize_column_nulls(df)
#           summarize_row_nulls(df)
#           prepare_zillow(df)
#           drop_missing_values(df, prop_required_column = 0, prop_required_row = 0)
#           feature_engineering(df)
#
#
################################################################################

import os
import pandas as pd

from sklearn.impute import SimpleImputer

from preprocessing import split_data

################################################################################

def summarize_column_nulls(df):
    return pd.concat([
        df.isnull().sum().rename('rows_missing'),
        df.isnull().mean().rename('percent_missing')
    ], axis = 1)

################################################################################

def summarize_row_nulls(df):
    return pd.concat([
        df.isnull().sum(axis = 1).rename('columns_missing'),
        df.isnull().mean(axis = 1).rename('percent_missing')
    ], axis = 1).value_counts().sort_index()

################################################################################

def prepare_zillow(df):
    df_copy = df.copy()
    df_copy = drop_missing_values(df_copy, prop_required_column = 0.8, prop_required_row = 1)
    df_copy = get_single_unit_properties(df)
    return df_copy

################################################################################

def drop_missing_values(df, prop_required_column = 0, prop_required_row = 0):
    df = df.dropna(axis = 'columns', thresh = round(df.shape[0] * prop_required_column))
    df = df.dropna(axis = 'index', thresh = round(df.shape[1] * prop_required_row))
    
    return df

################################################################################

def get_single_unit_properties(df):
    property_types = [
        'Single Family Residential',
        'Condominium',
        'Cluster Home',
        'Mobile Home',
        'Manufactured, Modular, Prefabricated Homes',
        'Residential General',
        'Townhouse'
    ]
    df = df[df.propertylandusedesc.isin(property_types)]
    
    return df

################################################################################

def feature_engineering(df):
    '''
    Feature engineering
    '''
    df['sqft_per_bed'] = df['calculatedfinishedsquarefeet'] / df['bedroomcnt']
    df['sqft_per_bath'] = df['calculatedfinishedsquarefeet'] / df['bathroomcnt']
    df['total_rooms'] = df['bedroomcnt'] + df['bathroomcnt']
    df['bed_bath_rooms_per_sqft_living'] = df['total_rooms'] / df['calculatedfinishedsquarefeet']
    df['age_in_years'] = 2017 - df['yearbuilt']
    df['taxrate'] = df['taxamount'] / df['taxvaluedollarcnt']
    df['dollars_per_sqft'] = df['taxvaluedollarcnt'] / df['calculatedfinishedsquarefeet']
    return df