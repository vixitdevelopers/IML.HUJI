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
    # todo:df = load_data("../datasets/City_Temperature.csv", parse_dates=['Date'])
    df = pd.read_csv("../datasets/City_Temperature.csv", parse_dates=['Date'])

    # remove all rows that the temp<-50 (this isn't possible, let's face it)
    df = df[df['Temp'] > -50]
    # create day_of_year column
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # Question 2 - Exploring data for specific country
    df_il = df[df['Country'] == 'Israel']

    # Define a palette of colors for each year
    palette = px.colors.qualitative.Alphabet
    # Create a scatter plot
    fig = px.scatter(df_il, x='DayOfYear', y='Temp', color='Year',
                     color_discrete_sequence=palette)
    fig.write_image("../israel_avg.png", engine='orca')

    df_il_monthly = df_il.groupby('Month')['Temp'].agg(['std']).reset_index()
    fig = px.bar(df_il_monthly, x='Month', y='std', title='Standard deviation compared monthly')
    fig.write_image('../deviation_monthly.png')

    # Question 3 - Exploring differences between countries

    df_country_monthly = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()
    # Create a line plot with error bars
    fig = px.line(df_country_monthly, x='Month', y='mean', error_y='std', color='Country',
                  title='Mean and deviation for each country by month')
    fig.write_image('../country_monthly.png')

    # Question 4 - Fitting model for different values of `k`

    # split the dataset
    # todo: do I split before or after I removed all null rows?
    x_train, y_train, x_test, y_test = split_train_test(df_il.DayOfYear, df_il.Temp)
    losses = []
    for k in range(1, 11):
        poly_model = PolynomialFitting(k)
        poly_model.fit(x_train, y_train)
        loss = round(poly_model.loss(X=x_test, y=y_test), 2)
        losses.append(loss)
        print(f'Loss for k={k} : {loss}')
    loss_df = pd.DataFrame({'k': list(range(1, 11)), 'loss': losses})
    fig = px.bar(loss_df, x='k', y='loss', title='MSE across different k chosen')
    fig.write_image('../poly_loss.png')

    CHOSEN_K = 7

    # Question 5 - Evaluating fitted model on different countries
    chosen_model = PolynomialFitting(CHOSEN_K)
    chosen_model.fit(x_train, y_train)

    df_jordan = df[df['Country'] == 'Jordan']
    df_nether = df[df['Country'] == 'The Netherlands']
    df_africa = df[df['Country'] == 'South Africa']
    country_losses = [chosen_model.loss(df_jordan.DayOfYear, df_jordan.Temp),
                      chosen_model.loss(df_nether.DayOfYear, df_nether.Temp),
                      chosen_model.loss(df_africa.DayOfYear, df_africa.Temp)]
    country_losses_df = pd.DataFrame({'Country': ['Jordan', 'The Netherlands', 'South Africa'],
                                      'loss': country_losses})
    fig = px.bar(country_losses_df, x='Country', y='loss',
                 title='MSE across different countries using Israel model (k=5)')
    fig.write_image('../countries_loss.png')
