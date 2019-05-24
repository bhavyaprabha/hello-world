import pandas as pd
from graphviz import Source
from IPython.display import display, SVG
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor


def plot_tree(train_features: pd.DataFrame, train_target: pd.DataFrame, feature_names: list,
              split: str, depth: int, min_split: float, min_leaf: float=0.2):
    """
    Interactive plotting for regression trees.
    Code adapted from https://towardsdatascience.com/interactive-visualization-of-decision-trees-with-jupyter-widgets-ca15dd312084
    :param train_features:
    :param train_target:
    :param feature_names:
    :param split:
    :param depth:
    :param min_split:
    :param min_leaf:
    :return:
    """
    model = DecisionTreeRegressor(random_state=0,
                                  splitter=split,
                                  max_depth=depth,
                                  min_samples_split=min_split,
                                  min_samples_leaf=min_leaf)
    model.fit(train_features, train_target)
    graph = Source(tree.export_graphviz(model,
                                        out_file=None,
                                        feature_names=feature_names,
                                        filled=True))
    display(SVG(graph.pipe(format='svg')))
    return model


def plot_open_and_closed_tickets(times: pd.DataFrame) -> None:
    """
    Plots the open and closed tickets per day
    :param times: Dataframe with the durations for each work item as computed by data_preparation.compute_work_item_times
    :return:
    """
    resample_period = 'D'  # We’re going to resample the dataframe per day
    open_per_day = times.resample(resample_period, on='start').work_item.count().rename('open_tickets_per_day')
    is_closed = times.end.notnull()
    closed_per_day = times.loc[is_closed]\
        .resample(resample_period, on='end')\
        .work_item.count()\
        .rename('closed_tickets_per_day')

    tickets_df = (pd.concat([open_per_day, closed_per_day], axis=1)
                    .fillna(0)
                    .astype(int)
                    .reset_index()
                 )

    tickets_df['open_tickets_total'] = tickets_df.open_tickets_per_day.cumsum()
    tickets_df['closed_tickets_total'] = tickets_df.closed_tickets_per_day.cumsum()
    tickets_df['wip_tickets_total'] = tickets_df.open_tickets_total - tickets_df.closed_tickets_total
    tickets_df.plot(x='start', y=['open_tickets_per_day', 'closed_tickets_per_day'],
                    figsize=(15, 10), subplots=True, alpha=0.5, sharey=True)
    return
