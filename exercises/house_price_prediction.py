from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
changed_counter = 0
averages = {}
train_columns = None


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # If it is the training set, remove all irrelevant rows (that are missing values):
    global averages, train_columns

    X_proc = pd.DataFrame.copy(X)
    X_proc['date'] = pd.to_numeric(X_proc['date'].str[:8], errors='coerce', downcast='integer')
    if y is None:
        for col in X_proc.columns:  # todo zipcode
            X_proc[col] = X_proc[col].replace(np.nan, averages.get(col, 0))
            X_proc[col] = X_proc[col].apply(replace_nonsensical_values, args=(col,))
    else:
        y_proc = pd.DataFrame.copy(y)
        X_proc = X_proc.dropna()
        y_proc = y_proc.loc[X_proc.index]
        averages = {col: X_proc[col].mean() for col in X_proc.columns}
        mask = pd.Series(True, index=X_proc.index)
        for col in X_proc.columns:
            mask &= sensical_values_indices(X_proc, col)
        X_proc = X_proc[mask]
        y_proc = y_proc[mask]

    X_proc['age'] = X_proc.apply(lambda x: x.date // 10_000 - x.yr_built, axis=1)
    X_proc['age_renovated'] = X_proc.apply(
        lambda x: x.age if x.yr_renovated == 0 else x.date // 10_000 - x.yr_renovated, axis=1)
    # dropping features with low correlation or which we switched:
    X_proc = X_proc.drop(['date', 'long', 'id', 'lat',
                          'sqft_lot15', 'yr_renovated', 'yr_built'], axis=1)
    X_proc['zipcode'] = X_proc['zipcode'].astype(int)
    X_proc = pd.get_dummies(X_proc, prefix='zipcode_', columns=['zipcode'])

    if y is None:
        X_proc = X_proc.reindex(columns=train_columns, fill_value=0)
    else:
        train_columns = X_proc.columns
        return X_proc, y_proc
    return X_proc[train_columns]


def replace_nonsensical_values(x, col_name):
    global averages, train_columns
    if np.isnan(x):
        return averages.get(col_name, 0)
    if col_name in ['sqft_living', 'sqft_avobe', 'yr_built'] and x <= 0:
        return averages.get(col_name, 0)
    if col_name in ['floors', 'sqft_basement', 'yr_renovated'] and x < 0:
        return averages.get(col_name, 0)
    if col_name == 'waterfront' and (x < 0 or x > 1):
        return averages.get(col_name, 0)
    if col_name == 'condition' and (x < 0 or x > 6):  # todo ceck
        return averages.get(col_name, 0)
    if col_name == 'grade' and (x < 0 or x > 16):
        return averages.get(col_name, 0)
    if col_name == 'bedrooms' and (x < 0 or x > 20):
        return averages.get(col_name, 0)
    if col_name == 'sqft_lot' and (x < 0 or x > 1_250_000):
        return averages.get(col_name, 0)
    # if col_name == 'zipcode' and f'zipcode_{x}' not in train_columns:
    #     return 98003  # this is just a random one
    return x


def sensical_values_indices(X: pd.DataFrame, col_name: str):
    if col_name in ['sqft_living', 'sqft_avobe', 'yr_built']:
        return X[col_name] > 0
    if col_name in ['floors', 'sqft_basement', 'yr_renovated']:
        return X[col_name] >= 0
    if col_name == 'waterfront':
        return X[col_name].isin(range(2))
    if col_name == 'view':
        return X[col_name].isin(range(5))
    if col_name == 'condition':
        return X[col_name].isin(range(1, 6))
    if col_name == 'grade':
        return X[col_name].isin(range(1, 16))
    if col_name == 'bedrooms':
        return X[col_name].isin(range(20))
    if col_name == 'sqft_lot':
        return X[col_name] > 0
    if col_name == 'long':
        return X[col_name] < 0
    return X[col_name] >= 0


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    cols = []
    coals = []
    for col in X.columns:
        cols.append(col)
        A = X[col].astype(float)
        pearson_A = y.cov(A) / (y.std() * A.std())
        coals.append(round(pearson_A, 2))
        both = pd.concat([A, y], axis=1)
        fig = px.scatter(both, x=col, y='price',
                         title=f'Correlation between {col} and response'
                               f'\nPearson correlation: {pearson_A}')
        fig.write_image(f'{output_path}_{col}.png')
    print(cols, '\n', coals)


if __name__ == '__main__':

    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")
    # any prices that are Nan are irrelevant (both train and test):
    df.dropna(subset=['price'], inplace=True)
    # split to X and y:
    df_response = df['price']
    df = df.drop('price', axis=1)

    # Question 1 - split data into train and test sets
    train_x, train_y, test_x, test_y = split_train_test(df, df_response)
    # Question 2 - Preprocessing of housing prices dataset
    train_x, train_y = preprocess_data(train_x, train_y)
    test_x = preprocess_data(test_x)

    # train_x, train_y = preprocess_data(train_x, train_y)
    # test_x = preprocess_data(test_x)

    # Question 3 - Feature evaluation with respect to response
    # feature_evaluation(train_x, train_y, output_path='../correlations/')

    # # Question 4 - Fit model over increasing percentages of the overall training data
    # # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    # #   1) Sample p% of the overall training data
    # #   2) Fit linear model (including intercept) over sampled set
    # #   3) Test fitted model over test set
    # #   4) Store average and variance of loss over test set
    # # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    #
    losses = np.zeros((len(range(10, 101)), 10))
    for percent in range(losses.shape[0]):
        print('percent: ', percent + 10)
        for sample in range(losses.shape[1]):
            X, y, _, _ = split_train_test(train_x, train_y, ((percent + 10) / 100))
            linear_model = LinearRegression(include_intercept=True)
            linear_model.fit(X, y)
            loss = linear_model.loss(test_x, test_y)
            losses[percent][sample] = loss

    loss_mean, loss_std = losses.mean(axis=1), losses.std(axis=1)
    percent = list(range(10, 101))
    res = pd.DataFrame({'percent': list(range(10, 101)), 'mean': loss_mean, 'std': loss_std})
    fig = go.Figure([go.Scatter(x=percent, y=loss_mean, mode="markers+lines", showlegend=False),
                     go.Scatter(x=percent, y=loss_mean - 2 * loss_std, fill=None, mode='lines',
                                line=dict(color="lightgrey"), showlegend=False),
                     go.Scatter(x=percent, y=loss_mean + 2 * loss_std, fill='tonexty', mode='lines',
                                line=dict(color="lightgrey"), showlegend=False),
                     ])
    fig.layout = go.Layout(xaxis='Percentage sampled',
                           yaxis='MSE',
                           title='MSE of fitted model as function of percentage of data fitted over')
    fig.write_image('res34.png')
    #
    #
    #
    #
    #
    ps = list(range(10, 101))
    results = np.zeros((len(ps), 10))
    for i, p in enumerate(ps):
        for j in range(results.shape[1]):
            _X = train_x.sample(frac=p / 100.0)
            _y = train_y.loc[_X.index]
            results[i, j] = LinearRegression(include_intercept=True).fit(_X, _y).loss(test_x, test_y)

    m, s = results.mean(axis=1), results.std(axis=1)
    fig = go.Figure([go.Scatter(x=ps, y=m - 2 * s, fill=None, mode="lines", line=dict(color="lightgrey")),
                     go.Scatter(x=ps, y=m + 2 * s, fill='tonexty', mode="lines", line=dict(color="lightgrey")),
                     go.Scatter(x=ps, y=m, mode="markers+lines", marker=dict(color="black"))],
                    layout=go.Layout(title="Test MSE as Function Of Training Size",
                                     xaxis=dict(title="Percentage of Training Set"),
                                     yaxis=dict(title="MSE Over Test Set"),
                                     showlegend=False))
    fig.write_image("res35.png")
    print('finish')
