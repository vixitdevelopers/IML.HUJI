from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    m = 1_000
    X = np.random.normal(mu, sigma, m)
    fit_uni = UnivariateGaussian().fit(X)
    print((round(fit_uni.mu_, 3), round(fit_uni.var_, 3)))

    # Question 2 - Empirically showing sample mean is consistent
    sizes = np.arange(10, 1010, 10)
    est_diff = []
    for size in sizes:
        fit_x = UnivariateGaussian().fit(X[:size + 1])
        est_diff.append(np.absolute(fit_x.mu_ - mu))
    go.Figure([go.Scatter(x=sizes, y=est_diff, mode='markers+lines', name=r'$\diff\$', showlegend=False),
               ],
              layout=go.Layout(title=r"$\text{(2) Absolute Distance between estimated and true expectation}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="m$|\mu - \hat\mu|$",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    fit_pdf = fit_uni.pdf(X)
    go.Figure([go.Scatter(x=X, y=fit_pdf, mode='markers', showlegend=False),
               ],
              layout=go.Layout(title=r"$\text{(3) Empirical PDF under fitted model }$",
                               xaxis_title="$X-\\text{fitted model}$",
                               yaxis_title="PDF",
                               )).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    m = 1_000
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, .2, 0, .5], [.2, 2, 0, 0], [0, 0, 1, 0], [.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mean=mu, cov=sigma, size=m)
    fit_X = MultivariateGaussian().fit(X)
    print(np.round(fit_X.mu_, 3))
    print(np.round(fit_X.cov_, 3))

    # Question 5 - Likelihood evaluation

    f_1 = np.linspace(-10, 10, 200)
    f_3 = np.linspace(-10, 10, 200)
    log_likelihood_matrix = np.zeros((200, 200))
    # calculate
    for i, f_1_i in enumerate(f_1):
        for j, f_3_j in enumerate(f_3):
            arr = np.array([f_1_i, 0, f_3_j, 0])
            log_likelihood_matrix[i, j] = MultivariateGaussian.log_likelihood(arr, sigma, X)

    # plot
    fig2 = go.Figure(go.Heatmap(x=f_1, y=f_3, z=log_likelihood_matrix),
                     layout=go.Layout(
                         title=r"$\text{(5) HeatMap of log-likelihood of Gaussian under Expectations}$",
                         xaxis_title="f3 values", yaxis_title="f1 values"))
    fig2.show()

    # Question 6 - Maximum likelihood
    arg_max_f_1, arg_max_f_3 = np.unravel_index(np.argmax(log_likelihood_matrix), log_likelihood_matrix.shape)
    print(np.round(f_1[arg_max_f_1], 3), np.round(f_3[arg_max_f_3], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
