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

    if y is None:
        X_proc = X_proc.drop('date', axis=1)
        for col in X_proc.columns:
            # if col == 'zipcode':
            #     X_proc[col] = X_proc[col].replace(np.nan, '98001')
            X_proc[col] = X_proc[col].replace(np.nan, averages.get(col, 0))
        # return X_proc
        # X_proc['date'] = pd.to_numeric(X_proc['date'].str[:8], errors='coerce', downcast='integer')
        # for col in X_proc.columns:
        #     X_proc[col] = X_proc[col].apply(replace_nonsensical_values, args=(col,))
    else:
        y_proc = pd.DataFrame.copy(y)
        X_proc = X_proc.dropna()
        y_proc = y_proc.loc[X_proc.index]
        X_proc = X_proc.drop('date', axis=1)
        # X_proc = X_proc.drop('zipcode', axis=1)
        averages = {col: X_proc[col].mean() for col in X_proc.columns}
        # return X_proc, y_proc
        #
        # X_proc['date'] = pd.to_numeric(X_proc['date'].str[:8])
        # mask = pd.Series(True, index=X_proc.index)
        # mask &= X_proc.notna().all(axis=1)
        # for col in X_proc.columns:
        #     mask &= sensical_values_indices(X_proc, col)
        # X_proc = X_proc[mask]
        # y_proc = y_proc[mask]

    # add age and age_renovated (which will just be the age if never renovated)
    # X_proc['age'] = X_proc.apply(lambda x: int(x.date / 10_000) - x.yr_built, axis=1)
    # X_proc['age_renovated'] = X_proc.apply(lambda x:
    #                                        int(x.date / 10_000) - x.yr_renovated if x.yr_renovated > 0 else x.age,
    #                                        axis=1)
    # dummies = pd.get_dummies(X_proc['zipcode'], prefix='zipcode_')
    # dummies['zipcode_other'] = 0
    # X_proc = X_proc.drop(["id", "lat", "long", "date", "sqft_lot15", "sqft_living15"], axis=1)

    X_proc = X_proc.drop(['long', 'id',
                          'sqft_lot15'], axis=1)

    # X_proc['zipcode'] = X_proc['zipcode'].astype(int)
    # X_proc['zipcode'] = X_proc['zipcode'].apply(lambda x: x // 10)
    X_proc = pd.get_dummies(X_proc, prefix='zipcode_', columns=['zipcode'])

    if y is None:
        X_proc = X_proc.reindex(columns=train_columns, fill_value=0)
        #
        # missing_cols = set(train_columns) - set(X_proc.columns)
        # for col in missing_cols:
        #     X_proc[col] = 0
        # other_cols = set(dummies) - set(train_dummies)
        # if other_cols:
        #     dummies['zipcode_other'] = (dummies[other_cols].isin([1]).any(axis=1)).astype(int)

    # X_proc = pd.concat([X_proc, dummies], axis=1)
    else:
        # train_dummies = dummies
        train_columns = X_proc.columns
        return X_proc, y_proc
    return X_proc[train_columns]


def replace_nonsensical_values(x, col_name):
    global averages, changed_counter

    if np.isnan(x):
        changed_counter += 1
        return averages.get(col_name) if averages.get(col_name) else 0

    if col_name == 'long':
        if x > 0:
            changed_counter += 1
            return averages.get(col_name) if averages.get(col_name) else 0
    else:
        if x < 0:
            changed_counter += 1
            return averages[col_name] if averages.get(col_name) else 0
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
    # # losses = np.zeros((len(range(10, 101)), 10))
    # # for percent in range(losses.shape[0]):
    # #     print('percent: ', percent + 10)
    # #     for sample in range(losses.shape[1]):
    # #         X, y, _, _ = split_train_test(train_x, train_y, ((percent + 10) / 100))
    # #         linear_model = LinearRegression(include_intercept=True)
    # #         linear_model.fit(X, y)
    # #         loss = linear_model.loss(test_x, test_y)
    # #         losses[percent][sample] = loss
    # #
    # # loss_mean, loss_std = losses.mean(axis=1), losses.std(axis=1)
    # # res = pd.DataFrame({'percent': list(range(10, 101)), 'mean': loss_mean, 'std': loss_std})
    # # fig = px.scatter(res, x='percent', y='mean')
    # # fig.write_image('res13.png')
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
    fig.write_image("mse.over.training.percentage.png")
    print('finish')
