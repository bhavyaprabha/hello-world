import pandas as pd
from nose.tools import assert_greater, eq_, raises
from src.evaluate import rmsle


def test_rmsle_equal():
    actuals_df = pd.DataFrame([['1', 1], ['2', 0]])
    actuals_df.columns = ['work_item', 'duration_in_days']
    predictions_df = pd.DataFrame([['1', 1], ['2', 0]])
    predictions_df.columns = ['work_item', 'predictions']
    val = rmsle(actuals_df, predictions_df)
    eq_(0, val)


def test_rmsle_not_equal():
    actuals_df = pd.DataFrame([['1', 1], ['2', 0]])
    actuals_df.columns = ['work_item', 'duration_in_days']
    predictions_df = pd.DataFrame([['1', 0], ['2', 0]])
    predictions_df.columns = ['work_item', 'predictions']
    val = rmsle(actuals_df, predictions_df)
    assert_greater(val, 0)


@raises(AssertionError)
def test_rmsle_different_work_items_nr1():
    actuals = pd.DataFrame([['1', 1], ['2', 2]])
    predictions = pd.DataFrame([['1', 1], ['2', 2], ['3', 3]])
    rmsle(actuals, predictions)


@raises(AssertionError)
def test_rmsle_different_work_items_nr2():
    predictions = pd.DataFrame([['1', 1], ['2', 2]])
    actuals = pd.DataFrame([['1', 1], ['2', 2], ['3', 3]])
    rmsle(actuals, predictions)


@raises(AssertionError)
def test_rmsle_different_work_items():
    actuals = pd.DataFrame([['1', 1], ['2', 2]])
    actuals.columns = ['work_item', 'duration_in_days']
    predictions = pd.DataFrame([['1', 1], ['3', 2]])
    predictions.columns = ['work_item', 'predictions']
    rmsle(actuals, predictions)
