import pandas as pd
import numpy as np

def show_open_tickets_on_date(df:pd.DataFrame , date:pd.Timestamp):
    """
    returns a data frame that includes information
    of open tickets on a given date
    df: pandas data frame object
    date: datetime 
    :return: pandas data frame object
    """
    try:
        df1 = df[df.timestamp == date]
        Open_per_date = df1[df1.Closed == 0]
    except KeyError as e:
        return print('The Dataframe must include {0} as a column'.format(e))
 
    return Open_per_date


def show_closed_tickets_before_date(df:pd.DataFrame , date:pd.Timestamp):
    """
    returns a data frame that includes information
    of closed tickets before a given date
    df: pandas data frame object
    date: datetime 
    :return: pandas data frame object
    """
    try:
        df1 = df[df.timestamp <= date]
        df_closed_tickets = df1[df1['Closed'] == 1]
    except KeyError as e:
        
        return print('The Dataframe must include {0} as a column'.format(e))
    return df_closed_tickets

def rmsle(Y_actual: np.array , Y_predict: np.array):
    """
    returns a float that represents the square
    root of the mean of the logarithmic 
    differences between predicted and actual 
    values. In general, a lower RMSLE is better
    than a higher one.
    Y_actual: Numpy array
    Y_predict: Numpy array
    :return: a float number
    """
    try:
        rmsle = np.sqrt(sum(((np.log(Y_predict + 1) - np.log(Y_actual + 1)) ** 2)) / len(Y_actual))
    except TypeError:
        return print('The inputs should be Numpy arrays')
    except ValueError:
        return print('The inputs should have equal length')
        
    return rmsle


def split_data(df: pd.DataFrame, target_name: str, size=0.6):
    """
    this function split the data frame
    with respect to the given size and
    provides four attributes
    x_train, y_train, x_test, y_test
    df: pandas data frame object
    target_name: string

    """
    try:
        nrow = int(len(df) * size)
        x_train = df.drop([target_name], axis = 1)[:nrow]
        y_train = df[target_name][:nrow]
        x_test = df.drop([target_name], axis = 1)[nrow:]
        y_test = df[target_name][nrow:]
        split_data.x_train = x_train
        split_data.y_train = y_train
        split_data.x_test = x_test
        split_data.y_test = y_test
    except (KeyError, TypeError):
        if KeyError:
            return print('second input must be a string')
        elif TypeError:
            return print('first input should be a dataframe')