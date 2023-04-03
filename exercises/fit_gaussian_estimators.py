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
    print((fit_uni.mu_, fit_uni.var_))

    # Question 2 - Empirically showing sample mean is consistent
    sizes = np.arange(10, 1010, 10)
    est_diff = []
    for size in sizes:
        fit_x = UnivariateGaussian().fit(X[:size + 1])
        est_diff.append(np.absolute(fit_x.mu_ - mu))
    go.Figure([go.Scatter(x=sizes, y=est_diff, mode='markers+lines', name=r'$\diff\$', showlegend=False),
               ],
              #
              layout=go.Layout(title=r"$\text{(2) Absolute Distance between estimated and true expectation}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="m$\|\mu - \hat\mu\|$",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    fit_pdf = fit_uni.pdf(X)
    go.Figure([go.Scatter(x=X, y=fit_pdf, mode='markers', showlegend=False),
               ],
              #
              layout=go.Layout(title=r"$\text{(2) Absolute Distance between estimated and true expectation}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="m$\|\mu - \hat\mu\|$",
                               )).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    m = 1_000
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([1, .2, 0, .5], [.2, 2, 0, 0], [0, 0, 1, 0], [.5, 0, 0, 1])
    X = np.random.multivariate_normal(mean=mu, cov=sigma, size=m)
    fit_X = MultivariateGaussian().fit(X)
    print('Mu_:', fit_X.mu_)
    print('Sigma_:', fit_X.cov_)

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    # test_multivariate_gaussian()
