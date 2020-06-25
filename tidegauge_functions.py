import os
import pandas as pd

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

    print(f'number of df.columns: {len(df.columns)}')
    print(f'length of columns: {len(columns)}')
    
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
