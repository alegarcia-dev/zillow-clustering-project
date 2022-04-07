################################################################################
#
#
#
#       clustering.py
#
#       Description: This file contains functions that are used for building 
#           cluster models.
#
#       Variables:
#
#           None
#
#       Functions:
#
#           plot_kmeans_inertia(df, columns, k_range)
#           create_clusters(df, columns, k)
#
#
################################################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

################################################################################

def plot_kmeans_inertia(df: pd.DataFrame, columns: list[str], k_range: tuple[int]) -> None:
    inertias = {}

    for k in range(k_range[0], k_range[1]):
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(df[columns])
        inertias[k] = kmeans.inertia_
        
    pd.Series(inertias).plot(xlabel = 'k', ylabel = 'Inertia')
    plt.show()

################################################################################

def create_clusters(df: pd.DataFrame, columns: list[str], k: int) -> pd.DataFrame:
    df_copy = df.copy()
    
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(df_copy[columns])

    df_copy['cluster'] = kmeans.predict(df_copy[columns])
    df_copy.cluster = df_copy.cluster.astype('category')

    return df_copy