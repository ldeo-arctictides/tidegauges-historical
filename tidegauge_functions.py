import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import glob
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


def calc_rolling_decomposition_GPS(df, period=365):

    decomposition_Vert = seasonal_decompose(df['Vertical'], period)

    df['trend_Vert'] = decomposition_Vert.trend
    df['seasonal_Vert'] = decomposition_Vert.seasonal
    df['residual_Vert'] = decomposition_Vert.resid

    decomposition_North = seasonal_decompose(df['North'], period)

    df['trend_North'] = decomposition_North.trend
    df['seasonal_North'] = decomposition_North.seasonal
    df['residual_North'] = decomposition_North.resid

    decomposition_East = seasonal_decompose(df['East'], period)

    df['trend_East'] = decomposition_East.trend
    df['seasonal_East'] = decomposition_East.seasonal
    df['residual_East'] = decomposition_East.resid

    return df


def read_GPS_SONEL(sonel_file, convert=True):
    i_skip = find_skiprows_startofline(sonel_file, '#  Year')
    column_names = ['Year', 'North', 'East', 'Vertical', 'NorthSTD', 'EastSTD', 'VerticalSTD']
    df = pd.read_csv(sonel_file, skiprows=i_skip, header=None, delimiter='\s+', names=column_names)
    year = df['Year'].astype(int)
    doy = ((df['Year'] - year) * 365).astype(int) + 1  # TODO: This might be off by one day...
    dt = pd.to_datetime(year.astype(str) + doy.astype(str), format='%Y%j')
    df = df.rename(columns={'Year': 'YearDec'})
    df.index = pd.DatetimeIndex(dt)
    
    return df


def calc_OLS_tides(df, var):
    x, y = np.arange(len(df[var].dropna())), df[var].dropna()
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    res = model.fit()
    
    return res


def convert_trend_toyearly(df, res):
    period = df.index.year.value_counts().max()
    yearlytrend = res.params.x1 * period
    
    return yearlytrend


def read_tidegauge_monthly(monthly_file):
    column_names=['Year', 'SSH', 'unknown1', 'unknown2']
    df = pd.read_csv(monthly_file, header=None, delimiter=';', names=column_names)
    
    # NaNs
    df['SSH'] = df['SSH'].replace(-99999, np.nan) 
    
    # Datetime operations
    year = df['Year'].astype(int)
    month = ((df['Year'] - year) * 12).astype(int) + 1
    dt = pd.to_datetime(year.astype(str) + month.astype(str), format='%Y%m')
    df = df.rename(columns={'Year': 'YearDec'})
    df.index = pd.DatetimeIndex(dt)

    return df


def read_CCAR_altimetry(ccar_file):
    i_skip = find_skiprows_startofline(ccar_file, 'time')
    column_names=['Year', 'SSH', '', 'Year2', 'SSH_tides']
    df = pd.read_csv(ccar_file, 
                     skiprows=i_skip,
                     header=None, 
                     delimiter=',', 
                     names=column_names)
    
    # NaNs
#     df['SSH'] = df['SSH'].replace(-99999, np.nan) 
    
    ## Datetime operations
    year = df['Year'].astype(int)
    doy = ((df['Year'] - year) * 365).astype(int) + 1  # TODO: This might be off by one day...
    dt = pd.to_datetime(year.astype(str) + doy.astype(str), format='%Y%j')
    df = df.rename(columns={'Year': 'YearDec'})
    df.index = pd.DatetimeIndex(dt)
    
    # delete unused columns
    df = df.drop([df.columns[2], 'Year2', 'SSH_tides'], axis=1)

    return df


def ADF_Summary(df1, df2):
    for df1, filepath in enumerate(df1):
        df1 = read_GPS_SONEL(filepath)
        files = print(f'\n\n{filepath}')
    
        result = adfuller(df1['Vertical'])
        #print(result)
        print('ADF Statistic: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            ADF = print('\t{}: {}'.format(key, value))
    
    for df2, filepath in enumerate(df2):
        df2 = read_GPS_nam14_UNAVCO(filepath)
        files = print(f'\n\n{filepath}')
    
        result = adfuller(df2['Vertical'])
        #print(result)
        print('ADF Statistic: {}'.format(result[0]))
        print('p-value: {}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            ADF = print('\t{}: {}'.format(key, value))
    return(files, ADF)


def plot_OLS_overlay(df, res, site, var, data_units, simpletrend=True):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(12,6));
    ax.plot(df[var].dropna().index, df[var].dropna().values, 
            label='Data', marker=',', linestyle='', color='black')
    
    ## Plot linea model
    if simpletrend:
        ax.plot((df[var].index[0], df[var].index[-1]), 
            (res.params.x1*1 + res.params.const, res.params.x1*df.shape[0] + res.params.const),
               label='Trend', linestyle='--', color='purple')
    else:
            ax.plot(df[var].index, [res.params.x1*i + res.params.const for i in np.arange(len(df[var]))])

    ## zero line
#     ax.plot((df[var].index[0], df[var].index[-1]), (0, 0), 'k')
    
    ## customize
#     ax.set_title(f"Trend = {res.params.x1 * 1000:.2f} mm/yr");
    ax.set_ylabel(data_units)
    plt.suptitle(f"{site}")
    plt.legend()
    plt.savefig(f'figs/test_GPS_OLS_{site}.png')