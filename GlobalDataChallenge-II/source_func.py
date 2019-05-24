import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import OneHotEncoder

def eval_model(model, train_features, train_times, test_features, test_times) -> (float,float):
    model.fit(train_features, train_times.duration_in_days)
    train_predictions = pd.DataFrame(train_times.work_item)  
    train_predictions['predictions'] = model.predict(train_features)
    test_predictions = pd.DataFrame(test_times.work_item)
    test_predictions['predictions'] = model.predict(test_features)
    train_rmsle = rmsle(train_times, train_predictions)
    test_rmsle = rmsle(test_times, test_predictions)
    return train_rmsle, test_rmsle



def orig_model(model, closed_features, closed_times, open_features, open_times) -> (float,float):
    model.fit(closed_features, closed_times.duration_in_days)
    closed_predictions = pd.DataFrame(closed_times.work_item)  
    closed_predictions['predictions'] = model.predict(closed_features)
    open_predictions = pd.DataFrame(open_times.work_item)
    open_predictions['predictions'] = model.predict(open_features)
    closed_rmsle = rmsle(closed_times, closed_predictions)
    open_rmsle = rmsle(open_times, open_predictions)
    with open('open_predictions_dtr.csv', 'w') as f:
        open_predictions.to_csv(f, index=False)
    return closed_rmsle, open_rmsle


def rmsle(actuals: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """
    Computes the root mean square log error between the actuals and predictions.
    Raises and error if there are multiple predictions for a single work item, or if there are missing predictions
    :param actuals: A DataFrame with the columns 'work_item' and 'duration_in_days'
    :param predictions: A DataFrame with the columns 'work_item' and 'predictions'
    :return: RMSLE between actuals and predictions
    """
    assert len(actuals) == len(predictions)
    assert set(actuals.work_item.values) == set(predictions.work_item.values)
    actuals_values = actuals.duration_in_days.values
    predictions_values = predictions.predictions.values
    rmsle = np.sqrt(sum(((np.log(actuals_values + 1) - np.log(predictions_values + 1)) ** 2)) / len(actuals_values))
    return rmsle

def compute_work_item_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame with the ticket data and computes the start time, end time,  duration and the
    duration_in_hours.
    :param df: As described above
    :return: As described above
    """
    # We can't be sure that NAs were already renamed. Do it again just to be sure.
    df.from_phase.fillna('Start', inplace=True)
    df.to_phase.fillna('End', inplace=True)
    df['components'].fillna('',inplace=True)
    relevant_columns = ['work_item', 'timestamp']
    start_times = df[df.from_phase == 'Start'][relevant_columns]
    end_times = df[df.to_phase == 'End'][relevant_columns]
    times = pd.merge(start_times, end_times, on='work_item', how='left')
    times.rename(columns={'timestamp_x': 'start', 'timestamp_y': 'end'}, inplace=True)
    times['duration'] = times['end'] - times['start']
    times['duration_in_days'] = times['duration'].apply(lambda x: round(x.total_seconds() / (24*3600), 2))
    return times

def split_times(times: pd.DataFrame, sep_date_str: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits the times DataFrame into three different DataFrames
    - train: closed before the sep_date
    - test:  closed after the sep_date and started before the sep_date
    - closed: tickets that are closed
    - open:  tickets that arent't closed yet
    :param times: DataFrames containing the start and end times for each work items.
    :param sep_date_str: Separation date as string in the format DD.MM.YYYY
    :return: A tuple of (train, test, open) as defined above
    """
    open_times = times[pd.isnull(times.duration)]
    closed_times = times[~pd.isnull(times.duration)]
    sep_date = dt.datetime.strptime(sep_date_str, '%d.%m.%Y')
    train_times = closed_times[closed_times.end <= sep_date]
    test_times = closed_times[(closed_times.end > sep_date) & (closed_times.start <= sep_date)]
    return train_times, test_times, closed_times, open_times

def tkts_per_day(times: pd.DataFrame) -> pd.DataFrame:
    open_per_day=times.resample('D',on='start')['work_item'].count().rename('open_tickets_per_day')
    is_closed = times['end'].notnull()
    closed_per_day = times.loc[is_closed].resample('D', on='end')['work_item'].count().rename('closed_tickets_per_day')
    tickets_df = (pd.concat([open_per_day, closed_per_day], axis=1) # Join the two dataframes
                    .fillna(0)                                      # Replace NaNs by 0 for those days when no tickets are opened or closed
                    .astype(int)                                    # While we're at it, all counts are integers
                    .reset_index()                                  # timestamp_x is used as index, move it back to a column
                    .rename(columns={'start': 'date'})        # and rename it to ‘date’
                 )
    tickets_df['open_tickets_total'] = tickets_df['open_tickets_per_day'].cumsum()
    tickets_df['closed_tickets_total'] = tickets_df['closed_tickets_per_day'].cumsum()
    tickets_df['wip_tickets_total'] = tickets_df['open_tickets_total'] - tickets_df['closed_tickets_total']
    return tickets_df

def onehot(train_df: pd.DataFrame,test_df: pd.DataFrame,cols_to_encode: list) -> (pd.DataFrame,pd.DataFrame):
    onehotencoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    # There will be components that only appear in the test set. To deal with this, we change the handle_unknown flag.
    train_features = onehotencoder.fit_transform(train_df[cols_to_encode])
    test_features = onehotencoder.transform(test_df[cols_to_encode])
    return train_features,test_features