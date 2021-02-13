import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as stat
from statsmodels.tsa.arima.model import ARIMA

def basic_cleaning():
    """
    Cleans the CPI data:
    1)Sets indexes to ISO3 Standards
    2)replaces spaces with "_"
    3)drops rank column (we can use other methods to get rank)

    :return: pandas dataframe with cleaned data
    """
    df = pd.read_csv('policy_growth/CPI_data.csv')
    df.set_index('ISO3', inplace=True)
    df.columns = df.columns.str.lower().str.replace(" ","_")
    df = df.loc[:, ~df.columns.str.startswith('rank')] #returns everything, take columns without "Rank"
    print("done cleaning")

    return df

def time_series_stuff(df,lst_of_countries,entrd_lag=3,get_acf = False, get_model=False):
    """
    Takes a dataframe and a list of interested countries
    shows a plot of the countries' CPI throughout time

    :param df: pandas dataframe
    :param lst_of_countries: list of ISO3 country codes
    :param entrd_lag: number of lags in the model
    :param get_acf: True if you want ACF/PACF graphs
    :param get_model: True if you want the AR models
                        for each country in lst_of_countries
    :return: if get_model, then returns the dictionary containing all models,
                if, get_acf, then plots the ACF/PACF plots
                else, returns the dataframe in time-series ready form
    """

    loc_df = df.loc[:, df.columns.str.startswith('cpi')]
    loc_df.columns = loc_df.columns.str.replace('cpi_score_', '')
    loc_df = loc_df.T
    loc_df[lst_of_countries].plot(kind='line')
    plt.legend(loc='best')
    plt.show()

    if get_acf == True:

        for country in lst_of_countries:
            fig = plt.figure(figsize=(12, 8))
            ax1 = fig.add_subplot(211)
            fig = stat.graphics.tsa.plot_acf(loc_df[country].values.squeeze(), lags=entrd_lag, ax=ax1)
            ax2 = fig.add_subplot(212)
            fig = stat.graphics.tsa.plot_pacf(loc_df[country], lags=entrd_lag, ax=ax2)
            plt.show()

    elif get_model == True:
        dict_of_models = {}
        for country in lst_of_countries:
            dict_of_models["{0}".format(country) + "_arma20"] = ARIMA(loc_df[country], order=(2, 0, 0)).fit()

        for pair in dict_of_models.items():
            print(pair[0])
            print(pair[1].params)

        return dict_of_models

    else:
        return loc_df


#Flow:

gdf = basic_cleaning()
print(gdf.head())

interesting_countries = ['USA','DNK','SWE','POL','BLR','RUS']
time_series = ['USA','RUS']
time_series_stuff(gdf,time_series,get_acf=False,entrd_lag=3,get_model=True)