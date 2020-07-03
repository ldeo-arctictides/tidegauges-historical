import os
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def find_skiprows_startofline(infile, search_string, diagnostics=False):
    '''

    Parameters
    ----------
    infile
    search_string
    diagnostics
    
    Returns
    -------
    i
    '''
    with open(infile) as fp:
        for i, line in enumerate(fp):
            if line[:len(search_string)] == search_string:
                if diagnostics:
                    print(line)
                    print(len(search_string))
                return i + 1


def read_tidegauge_psmsl(path, columns=None):
    """
    Read and parse PSMSL tide gauge data

    CSV Columns:
        'YEAR', 'MONTH', 'DAY', 'SSH', possibly 'HOUR'

    Parameters
    ----------
    path : str
        Filesystem path to data file
    columns: List

    Returns
    -------
    pandas.DataFrame
        Gravity data indexed by datetime.
    """
    columns = ['YEAR', 'MONTH', 'DAY', 'SSH']

    df = pd.read_csv(path, header=None, engine='c')

    if len(df.columns) == len(columns):
        #         columns += ['unknown']
        df.columns = ['YEAR', 'MONTH', 'DAY', 'SSH']
    else:
        df.columns = ['YEAR', 'MONTH', 'DAY', 'HOUR', 'SSH']

    # missing data to NaNs
    df['SSH'] = df['SSH'].replace(-32767, np.nan)

    # create datetime index
    dt = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    df.index = pd.DatetimeIndex(dt)

    return df


def read_GPS_nam14_UNAVCO(path, columns=None):
    columns = ['Date', 'North', 'East', 'Vertical', 'NorthSTD', 'EastSTD', 'VerticalSTD', 'Quality', 'NaN']
    df = pd.read_csv(path, skiprows=12, header=None, engine='c')
    if len(df.columns) == len(columns):
        df.columns = ['Date', 'North', 'East', 'Vertical', 'NorthSTD', 'EastSTD', 'VerticalSTD', 'Quality', 'NaN']
    dt = pd.to_datetime(df['Date'])

    # create datetime index
    df.index = pd.DatetimeIndex(dt)

    # Clean up
    df = df.drop('Date', axis=1)
    df = df.drop('NaN', axis=1)
    
    return df


def calc_rolling_decomposition_GPS(df):

    decomposition_Vert = seasonal_decompose(df['Vertical'], period=365)

    df['trend_Vert'] = decomposition_Vert.trend
    df['seasonal_Vert'] = decomposition_Vert.seasonal
    df['residual_Vert'] = decomposition_Vert.resid

    decomposition_North = seasonal_decompose(df['North'], period=365)

    df['trend_North'] = decomposition_North.trend
    df['seasonal_North'] = decomposition_North.seasonal
    df['residual_North'] = decomposition_North.resid

    decomposition_East = seasonal_decompose(df['East'], period=365)

    df['trend_East'] = decomposition_East.trend
    df['seasonal_East'] = decomposition_East.seasonal
    df['residual_East'] = decomposition_East.resid

    return df


def read_GPS_SONEL(sonel_file):
    i_skip = find_skiprows_startofline(sonel_file, '#  Year')
    column_names = ['Year', 'DN', 'DE', 'DU', 'SDN', 'SDE', 'SDU']
    df = pd.read_csv(sonel_file, skiprows=i_skip, header=None, delimiter='\s+', names=column_names)
    year = df['Year'].astype(int)
    doy = ((df['Year'] - year) * 365).astype(int) + 1  # TODO: This might be off by one day...
    dt = pd.to_datetime(year.astype(str) + doy.astype(str), format='%Y%j')
    df = df.rename(columns={'Year': 'YearDec'})
    df.index = pd.DatetimeIndex(dt)

    return df
