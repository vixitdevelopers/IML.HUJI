import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = pd.read_csv("../datasets/City_Temperature.csv", parse_dates=True)

    # remove all rows that the temp<-50 (this isn't possible, let's face it)
    df = df[df['Temp'] > -50]
    # create day_of_year column
    df['DayOfYear'] = df['date'].dt.dayOfYear

    # Question 2 - Exploring data for specific country
    df_il = df[df['Country'] == 'Israel']

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
