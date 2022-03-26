"""
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib import dates
from typing import Iterable, Callable


def date_to_datetime(date: str) -> datetime.datetime:
    """
    Convert a date string into a datetime object.

    If the date string contains hours, minutes, seconds, these are
    included. Milliseconds are ignored.

    Accepted formats: YYYY-MM-DD HH:MM:SS.000 where 000 are ms.
                      YYYY-MM-DD

    :param date: A string representing a date in one of the accepted
                 formats.
    :type date: str

    :returns: A datetime object representing the date.
    :rtype: datetime.datetime
    """
    try:
        return datetime.datetime.strptime(date[:-4], "%Y-%m-%d %H:%M:%S")
    except(ValueError):
        return datetime.datetime.strptime(date, "%Y-%m-%d")

def moving_average(data_array: Iterable, window: int) -> np.array:
    """
    Calculate a moving average by convolving along a dataset.

    :param data_array: Data to average over.
    :type data_array: Iterable
    :param window: Window size to average over.
    :type window: int

    :returns: The averaged data, in the same size array.
    :rtype: np.array
    """
    return np.convolve(data_array, np.ones((window,))/window, mode="same")

def binarise(data_array: Iterable, value) -> np.array:
    """
    Convert all elements of data_array that are equal to value to 1,
    convert the rest to 0.

    :param data_array: Data to binarise.
    :type data_array: Iterable
    :param value: Value to check against.
    :type value: Any.

    :returns: A data array of 1s and 0s representing whether each point
              is equal to value.
    :rtype: np.array
    """
    return np.array([int(x==value) for x in data_array])

def binarise_df(dataframe: pd.DataFrame,
                value: str,
                column: str) -> pd.DataFrame:
    """
    Apply binarise to a pd.DataFrame object.

    Creates a new column titled value with the binarised data.

    :param dataframe: Data to binarise.
    :type dataframe: pd.DataFrame
    :param value: Value to check against.
    :type value: Any.
    :param column: Column of the dataframe to binarise.
    :type column: str

    :returns: A dataframe with a new column titled value containing the
              binarised data.
    :rtype: pd.DataFrame
    """
    dataframe[value] = binarise(dataframe[column], value)
    return dataframe

def add_day(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column to a dataframe containing only the day of the date.

    :param dataframe: Dataframe to calculate data for.
    :type dataframe: pd.DataFrame

    :returns: The same dataframe inputted with a new column for "day"
    :rtype: pd.DataFrame
    """
    dataframe['day'] = dataframe['date'].str.slice(0, 10)
    return dataframe

def plot_percentage_by_date(data: pd.DataFrame, window: int=10000) -> None:
    """
    Visualise a moving average of each interaction type over time.

    Requires a dataframe with column "status" containing values "Opened",
    "Error", "Clicked", "Unsubscribed". Requires a column "date" in the
    format YYYY-MM-DD HH:MM:SS.000.

    :param data: Data to plot.
    :type data: pd.DataFrame
    :param window: Window over which to calculate a moving average.
    :type window: int

    :returns: None
    :rtype: None
    """
    data = add_day(data)
    data = data.sort_values(by='day')
    opens  = moving_average(binarise_df(data, "Opened"      , "status").groupby("day", as_index=False)['Opened'].mean()['Opened'], window)
    errors = moving_average(binarise_df(data, "Error"       , "status").groupby("day", as_index=False)['Error'].mean()['Error'], window)
    clicks = moving_average(binarise_df(data, "Clicked"     , "status").groupby("day", as_index=False)['Clicked'].mean()['Clicked'], window)
    unsubs = moving_average(binarise_df(data, "Unsubscribed", "status").groupby("day", as_index=False)['Unsubscribed'].mean()['Unsubscribed'], window)

    time = sorted(dates.date2num([date_to_datetime(d) for d in set(data['day'])]))

    plt.plot_date(time, opens, label="Opens", fmt='-')
    plt.plot_date(time, errors, label="Errors", fmt='-')
    plt.plot_date(time, clicks, label="Clicks", fmt='-')
    plt.plot_date(time, unsubs, label="Unsubs", fmt='-')
    
    plt.legend()
    plt.show()

def get_interactions(data: pd.DataFrame, interaction_type: str, window: int=10000) -> np.array:
    """
    Get a moving average of an interaction type over time.

    :param data: Data to calculate the average for.
    :type data: pd.DataFrame
    :param interaction_type: Interaction type to view: Opened, Error,
                             Clicked, Unsubscribed
    :type interaction_type: str
    :param window: WIndow over which to calculate a moving average.
    :type window: int

    :returns: An array of the moving average of the given interaction
              type over time.
    :rtype: np.array
    """
    data = add_day(data)
    data = data.sort_values(by='day')
    return moving_average(binarise_df(data, interaction_type, "status").groupby("day", as_index=False)[interaction_type].mean()[interaction_type], window)

def get_times(data: pd.DataFrame) -> np.array:
    """
    Convert the date column of a dataframe into a plottable array of
    integers.

    :param data: Data to convert
    :type data: pd.DataFrame

    :returns: An array of integers representing dates.
    :rtype: np.array
    """
    data = add_day(data)
    data = data.sort_values(by='day')
    return sorted(dates.date2num([date_to_datetime(d) for d in set(data['day'])]))
