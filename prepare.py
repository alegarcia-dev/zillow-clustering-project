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
from clustering import create_clusters

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
    df_copy = get_single_unit_properties(df_copy)

    df_copy['property_age'] = 2017 - df_copy['yearbuilt']
    df_copy = create_zip_code_bins(df_copy)

    columns = [
        'property_age',
        'calculatedfinishedsquarefeet',
        'lotsizesquarefeet'
    ]
    k = 4
    df_copy = create_clusters(df_copy, columns, k)
    df_copy = pd.get_dummies(df_copy, columns = ['cluster'])
    
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

################################################################################

def create_zip_code_bins(df):
    zip_codes = [
        96095.0, 96985.0, 96522.0, 96045.0, 96415.0, 96152.0, 96190.0, 96974.0, 
        96289.0, 96026.0, 96517.0, 96280.0, 96201.0, 96336.0, 96212.0, 95997.0, 
        96029.0, 96271.0, 96123.0, 97298.0, 97026.0, 96006.0, 96294.0, 96508.0, 
        96437.0, 96047.0, 96507.0, 96217.0, 96426.0, 96514.0, 95989.0, 96020.0, 
        96022.0, 96326.0, 96127.0, 96005.0, 96120.0, 96379.0, 96234.0, 95984.0, 
        96016.0, 96240.0, 96017.0, 96103.0, 97084.0, 96097.0, 96137.0, 96043.0, 
        96136.0, 96134.0, 96216.0
    ]

    df_copy = df.copy()
    df_copy['non_average_zip_code'] = df_copy.regionidzip.isin(zip_codes)
    df_copy.non_average_zip_code.astype = df_copy.non_average_zip_code.astype('int')

    return df_copy