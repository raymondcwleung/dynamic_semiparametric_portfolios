import jax.numpy as jnp
import jax.scipy as jscipy
import jax.scipy.optimize
from jax import grad, jit, vmap
from jax import random
import jax
import jax.test_util

import typing as tp

from jax import Array
from jax.typing import ArrayLike, DTypeLike
from jax._src.random import KeyArrayLike
from numpy.ma.core import identity

import optax
import jaxopt

import numpy as np
import scipy
import matplotlib.pyplot as plt


import time

# HACK:
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod
jax.config.update("jax_debug_nans", True)  # Should disable in full prod


def _pdf_normalized_skew_normal(z: ArrayLike, alpha: float) -> Array:
    """
    Density of a normalized skew-normal random variable.
    See Azzalini (2014) Section 2.1.
    """
    return 2 * jscipy.stats.norm.pdf(z) * jscipy.stats.norm.cdf(alpha * z)


def pdf_skew_normal(y: ArrayLike, xi: float, omega: float, alpha: float) -> Array:
    """
    Density of a skew-normal random variable Y \\sim SN(\\xi, \\omega^2, \\alpha),
    where Y = \\xi + \\omega Z, and where Z is a normalized skew-normal
    random variable.
    """
    z = (y - xi) / omega
    return (1 / omega) * _pdf_normalized_skew_normal(z, alpha)


def pdf_standardized_skew_normal(
    x: ArrayLike, xi: float, omega: float, alpha: float
) -> Array:
    """
    Return the density of X = (Y - \\mu) / \\sigma, where
    Y \\sim SN(\\xi, \\omega^2, \\alpha), and where we set
    \\mu = \\mu(\\alpha) = E[Y], and \\sigma = \\sigma(\\alpha) =
    Var(Y).
    """
    # Compute the mean of Y
    mu_Y = mean_skew_normal(xi=xi, omega=omega, alpha=alpha)

    # Compute the variance of Y
    sigma2_Y = variance_skew_normal(omega=omega, alpha=alpha)
    sigma_Y = jnp.sqrt(sigma2_Y)

    # Return the pdf of X = (Y - \mu) / \sigma
    return sigma_Y * pdf_skew_normal(
        mu_Y + sigma_Y * x, xi=xi, omega=omega, alpha=alpha
    )


def pdf_mvar_indp_skew_normal(x, vec_xi, vec_omega, vec_alpha):
    """
    Density of a vector of independent (but not necessarily
    identically distributed) skew-normal random variables.
    """
    _func = vmap(pdf_skew_normal, in_axes=[0, 0, 0, 0])
    vec_pdf = _func(x, vec_xi, vec_omega, vec_alpha)

    return jnp.prod(vec_pdf)


def pdf_mvar_indp_standardized_skew_normal(x, vec_xi, vec_omega, vec_alpha):
    """
    Density of a vector of independent (but not necessarily
    identically distributed) standardized skew-normal
    random variables.
    """
    _func = vmap(pdf_standardized_skew_normal, in_axes=[0, 0, 0, 0])
    vec_pdf = _func(x, vec_xi, vec_omega, vec_alpha)

    return jnp.prod(vec_pdf)


def loglik_mvar_indp_skew_normal(data, vec_xi, vec_omega, vec_alpha):
    _pdf = lambda x: pdf_mvar_indp_skew_normal(x, vec_xi, vec_omega, vec_alpha)
    _func = vmap(_pdf, in_axes=0)

    summands = _func(data)
    loglik_summands = jnp.log(summands)

    loglik = loglik_summands.mean()
    return -1.0 * loglik


def loglik_mvar_indp_standardized_skew_normal(data, vec_xi, vec_omega, vec_alpha):
    _pdf = lambda x: pdf_mvar_indp_standardized_skew_normal(
        x, vec_xi, vec_omega, vec_alpha
    )
    _func = vmap(_pdf, in_axes=0)

    summands = _func(data)
    loglik_summands = jnp.log(summands)

    loglik = loglik_summands.mean()
    return -1.0 * loglik


def sample_skew_normal(
    key: KeyArrayLike, xi: ArrayLike, omega: ArrayLike, alpha: ArrayLike
) -> Array:
    """
    Sample a skew-normal random variable Y \\sim SN(\\xi, \\omega^2, \\alpha).
    Use the stochastic representation of Azzalini (2014) Section 2.1.3.
    """
    # X_0 and U are independent N(0,1) variables
    mean = jnp.repeat(0.0, 2)
    cov = jnp.identity(2)
    _ = jax.random.multivariate_normal(key=key, mean=mean, cov=cov)
    x0 = _[0]
    u = _[1]

    # Construct Z
    sgn = jnp.sign(alpha * x0 - u)
    z = sgn * x0

    # Construct Y = \xi + \omega Z
    y = xi + omega * z
    return y


def sample_standardized_skew_normal(
    key: KeyArrayLike, xi: ArrayLike, omega: ArrayLike, alpha: ArrayLike
) -> Array:
    """
    Sample a standardized skew-normal random variable X = (Y - \\mu) / \\sigma
    Y \\sim SN(\\xi, \\omega^2, \\alpha)
    """
    # Mean of Y
    mu_Y = mean_skew_normal(xi=xi, omega=omega, alpha=alpha)

    # Variance of Y
    sigma2_Y = variance_skew_normal(omega=omega, alpha=alpha)
    sigma_Y = jnp.sqrt(sigma2_Y)

    y = sample_skew_normal(key=key, xi=xi, omega=omega, alpha=alpha)
    x = (y - mu_Y) / sigma_Y
    return x


def _one_sample_mvar_indp_skew_normal(
    key: KeyArrayLike,
    vec_xi: Array,
    vec_omega: Array,
    vec_alpha: Array,
) -> Array:
    """ """
    dim = jnp.size(vec_xi)

    subkeys = random.split(key, dim)
    _func = vmap(sample_skew_normal, in_axes=[0, 0, 0, 0])
    return _func(subkeys, vec_xi, vec_omega, vec_alpha)


def _one_sample_mvar_indp_standardized_skew_normal(
    key: KeyArrayLike,
    vec_xi: Array,
    vec_omega: Array,
    vec_alpha: Array,
) -> Array:
    """ """
    dim = jnp.size(vec_alpha)

    subkeys = random.split(key, dim)
    _func = vmap(sample_standardized_skew_normal, in_axes=[0, 0, 0, 0])
    return _func(subkeys, vec_xi, vec_omega, vec_alpha)


def sample_mvar_indp_skew_normal(
    key: KeyArrayLike,
    num_sample: int,
    vec_xi: Array,
    vec_omega: Array,
    vec_alpha: Array,
) -> Array:
    """ """
    subkeys = random.split(key, num_sample)
    subkeys = jnp.array(subkeys)

    _func = vmap(_one_sample_mvar_indp_skew_normal, in_axes=[0, None, None, None])

    sample = _func(subkeys, vec_xi, vec_omega, vec_alpha)
    return sample


def sample_mvar_indp_standardized_skew_normal(
    key: KeyArrayLike,
    num_sample: int,
    vec_xi: Array,
    vec_omega: Array,
    vec_alpha: Array,
) -> Array:
    """ """
    subkeys = random.split(key, num_sample)
    subkeys = jnp.array(subkeys)

    _func = vmap(
        _one_sample_mvar_indp_standardized_skew_normal, in_axes=[0, None, None, None]
    )
    sample = _func(subkeys, vec_xi, vec_omega, vec_alpha)
    return sample


def _mean_normalized_skew_normal(alpha: ArrayLike) -> Array:
    """
    Compute the mean of Z \\sim SN(0, 1, \\alpha).
    See Azzalini (2014) Section 2.1
    """
    sqrt = jnp.sqrt
    pi = jnp.pi

    b = sqrt(2 / pi)
    delta = alpha / sqrt(1 + alpha**2)

    # \mu_Z
    mu_Z = b * delta
    return mu_Z


def _var_normalized_skew_normal(alpha: ArrayLike) -> Array:
    """
    Compute the variance of Z \\sim SN(0, 1, \\alpha).
    See Azzalini (2014) Section 2.1
    """
    mu_Z = _mean_normalized_skew_normal(alpha=alpha)

    # \sigma_Z^2
    sigma2_Z = 1 - mu_Z**2
    return sigma2_Z


def mean_skew_normal(xi: ArrayLike, omega: ArrayLike, alpha: ArrayLike) -> Array:
    """
    Compute the mean of Y \\sim SN(\\xi, \\omega^2, \\alpha).
    See Azzalini (2014) Section 2.1
    """
    mu_Z = _mean_normalized_skew_normal(alpha=alpha)
    mu_Y = xi + omega * mu_Z
    return mu_Y


def variance_skew_normal(omega: ArrayLike, alpha: ArrayLike) -> Array:
    """
    Compute the variance of Y \\sim SN(\\xi, \\omega^2, \\alpha).
    See Azzalini (2014) Section 2.1
    """
    sigma2_Z = _var_normalized_skew_normal(alpha=alpha)

    sigma2_Y = omega**2 * sigma2_Z
    return sigma2_Y


def mean_mvar_indp_skew_normal(
    vec_xi: ArrayLike, vec_omega: ArrayLike, vec_alpha: ArrayLike
) -> Array:
    _func = vmap(mean_skew_normal, in_axes=[0, 0, 0])
    vec_means = _func(vec_xi, vec_omega, vec_alpha)
    return vec_means


def variance_mvar_indp_skew_normal(vec_omega: ArrayLike, vec_alpha: ArrayLike) -> Array:
    _func = vmap(variance_skew_normal, in_axes=[0, 0])
    vec_variance = _func(vec_omega, vec_alpha)
    return vec_variance


def pdf_mvar_standardized_indp_skew_normal(vec_alpha):
    """
    Density of standardized multivariate indepndent skew normal random
    variables.

    In particular, let Y_i \\sim SN(\\xi_i, \\omega_i^2, \\alpha_i).
    Set \\xi_i = -E[Y_i] so that \E[Y_i] = 0. Se
    """


key = random.key(123457)

num_sample = int(5e4)

dim = 1
x = (1 / 2) * jax.random.uniform(key, shape=(dim,)) - (1 / 4)
key, subkey = random.split(key)
vec_xi_true = (1 / 2) * jax.random.uniform(key, shape=(dim,)) - (1 / 4)
key, subkey = random.split(key)
vec_omega_true = jnp.sqrt((1 / 3) * jax.random.uniform(key, shape=(dim,)) ** 2)
key, subkey = random.split(key)
vec_alpha_true = (1 / 2) * jax.random.uniform(key, shape=(dim,)) - (1 / 4)

vec_xi_true = jnp.repeat(0.0, dim)
vec_omega_true = jnp.repeat(1.0, dim)


# data = sample_mvar_indp_skew_normal(
#     key,
#     num_sample=num_sample,
#     vec_xi=vec_xi_true,
#     vec_omega=vec_omega_true,
#     vec_alpha=vec_alpha_true,
# )
#
# sample_vec_means = jnp.mean(data, axis=0)
# sample_cov = jnp.cov(data, rowvar=False)


data = sample_mvar_indp_standardized_skew_normal(
    key,
    num_sample=num_sample,
    vec_xi=vec_xi_true,
    vec_omega=vec_omega_true,
    vec_alpha=vec_alpha_true,
)

sample_vec_means = jnp.mean(data, axis=0)
sample_cov = jnp.cov(data, rowvar=False)


def objfun_unstd(x, data):
    dim = data.shape[1]

    vec_xi = x[0:dim]
    vec_omega = x[dim : (2 * dim)]
    vec_alpha = x[(2 * dim) :]

    # vec_xi = x[0:dim]
    # vec_alpha = x[dim:]

    return loglik_mvar_indp_skew_normal(
        data=data,
        vec_xi=vec_xi,
        vec_omega=vec_omega,
        vec_alpha=vec_alpha,
    )


num_params = 3
options = {"maxiter": 1000, "gtol": 1e-3}
key, subkey = random.split(key)
x0 = (1 / 2) * jax.random.uniform(key, shape=(dim * num_params,)) - (1 / 4)
x0 = x0.at[1].set(np.abs(x0[1]))
optres_unstd = jscipy.optimize.minimize(
    objfun_unstd,
    x0=x0,
    method="BFGS",
    args=(data,),
    options=options,
)

breakpoint()


pop_vec_means = mean_mvar_indp_skew_normal(
    vec_xi=vec_xi_true, vec_omega=vec_omega_true, vec_alpha=vec_alpha_true
)
pop_vec_variance = variance_mvar_indp_skew_normal(
    vec_omega=vec_omega_true, vec_alpha=vec_alpha_true
)
pop_vec_std = jnp.sqrt(pop_vec_variance)


def objfun(x, data):
    dim = data.shape[1]
    vec_alpha = x

    # Data is such that X = (Y - \mu_Y) / \sigma_Y
    # where \mu_Y = \mu_Y(\alpha) and
    # \sigma_Y = \sigma_Y(\alpha)
    # and Y \\sim SN(0, 1, \alpha)

    vec_xi = jnp.repeat(0.0, dim)
    vec_omega = jnp.repeat(1.0, dim)

    # Mean of Y
    mu_Y = mean_mvar_indp_skew_normal(
        vec_xi=vec_xi, vec_omega=vec_omega, vec_alpha=vec_alpha
    )

    # Variance of Y
    sigma2_Y = variance_mvar_indp_skew_normal(vec_omega=vec_omega, vec_alpha=vec_alpha)
    sigma_Y = jnp.sqrt(sigma2_Y)

    # Need to "unstandardize" the data
    unstd_data = data * sigma_Y + mu_Y

    res = loglik_mvar_indp_skew_normal(
        data=unstd_data,
        vec_xi=vec_xi,
        vec_omega=vec_omega,
        vec_alpha=vec_alpha,
    )

    # vec_xi = jnp.repeat(0.0, dim)
    # vec_omega = jnp.repeat(1.0, dim)
    #
    # res = loglik_mvar_indp_standardized_skew_normal(
    #     data=data, vec_xi=vec_xi, vec_omega=vec_omega, vec_alpha=vec_alpha
    # )
    return res


# # blah = lambda x: pdf_skew_normal(
# #     x, xi=vec_xi_true, omega=vec_omega_true, alpha=vec_alpha_true
# # )
# # blah = lambda x: pdf_standardized_skew_normal(x, alpha=vec_alpha_true)
# blah = lambda x: objfun(x=jnp.array([x]), data=data)
# xx = jnp.linspace(-1, 1, 50)
# yy = [blah(x) for x in xx]
# yy = jnp.array(yy)
# plt.plot(xx, yy)
# plt.show()


num_params = 1

key, subkey = random.split(key)
x0 = (1 / 2) * jax.random.uniform(key, shape=(dim * num_params,)) - (1 / 4)

# optres = jscipy.optimize.minimize(
#     objfun, x0=x0, method="BFGS", args=(data, vec_xi_true, vec_omega_true)
# )

options = {"maxiter": 3000, "gtol": 1e-3}
optres = jscipy.optimize.minimize(
    objfun, x0=x0, method="BFGS", args=(data,), options=options
)
