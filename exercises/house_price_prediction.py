from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


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
    # add age and age_renovated (which will just be the age if never renovated)
    X['age'] = X.apply(lambda x: int(x.date[0:4]) - x.yr_built, axis=1)
    X['age_renovated'] = X.apply(lambda x:
                                 int(x.date[0:4]) - x.yr_renovated if x.yr_renovated > 0 else x.age, axis=1)
    # convert date to pandas date_time
    X['date'] = pd.to_datetime(X['date'], format='%Y%m%dT%H%M%S').dt.date

    # keep only first 4 digits of zipcode
    X['zipcode'] = X['zipcode'].str[:4]

    for col in X.columns:
        X[col] = X[col].apply(replace_nonsensical_values, args=(col,))

    # create dummy variables from the 'zipcode' column


#     todo create dummies


def replace_nonsensical_values(x, col_name):
    if col_name == 'long':
        if x > 0:
            return -122  # todo save averages
    else:
        if x < 0:
            return 0  # todo save averages


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

    A = X["id"]
    pearson_A = y.cov(A) / (y.std() * A.std())

    # pearson_correlation = lambda a, b: a.cov(b) / (a.std() * b.std())


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")
    df_response = df['price']
    df = df.drop('price', axis=1)
    # Question 1 - split data into train and test sets
    train_x, train_y, test_x, test_y = split_train_test(df, df_response)

    # Question 2 - Preprocessing of housing prices dataset
    # raise NotImplementedError()

    # Question 3 - Feature evaluation with respect to response
    # raise NotImplementedError()

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    for percent in range(10, 101):
        for _ in range(10):
            X, _, y, _ = split_train_test(train_x, train_y, (percent / 100))  # todo: randomly
            linear_model = LinearRegression()
            linear_model.fit(X, y)
            loss = linear_model.loss(test_x, test_y)
            loss_mean, loss_std = pd.Series(loss).mean(), pd.Series(loss).std()
