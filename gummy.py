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

import optax
import jaxopt

import numpy as np
import scipy
import matplotlib.pyplot as plt


import time

from scipy.linalg import cholesky

# HACK:
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
# jax.config.update("jax_enable_x64", True) # Should use x64 in full prod
jax.config.update("jax_debug_nans", True)  # Should disable in full prod


def calc_psd_matrix_invsqrt(mat_X: ArrayLike) -> Array:
    """
    Compute the inverse square root of positive definite symmetric matrix X.
    That is, return the matrix B such that B^2 = B^{\\top} B = X^{-1}.
    In other words, use the notation X^{-1/2} := B.
    """
    diag = jnp.diag
    eigh = jnp.linalg.eigh
    transpose = jnp.transpose

    # Use the spectral theorem on the fuction t \mapsto t^{-1/2}
    eigenvals, eigenvecs = eigh(mat_X)
    mat_inv_sqrt_eigenvals = diag(eigenvals ** (-1 / 2))

    # Compute X^{-1/2}
    mat_X_invsqrt = eigenvecs @ mat_inv_sqrt_eigenvals @ transpose(eigenvecs)

    return mat_X_invsqrt


def _pdf_normalized_multivariate_skew_normal(
    x: ArrayLike, mat_omega_bar: ArrayLike, vec_alpha: ArrayLike
) -> Array:
    """
    Density of a SN_d(0, \\bar{\\Omega}, \\alpha) random vector.
    See Azzalini (2014, Chapter 5), eq (5.1).
    """
    inner = jnp.inner

    dim = jnp.size(x)

    # \phi_d(x ; \Omega)
    phi_d = jscipy.stats.multivariate_normal.pdf(
        x=x, mean=jnp.repeat(0, dim), cov=mat_omega_bar
    )
    # \Phi(\alpha^\top x)
    phi = jscipy.stats.norm.cdf(x=inner(vec_alpha, x))

    return 2 * phi_d * phi


def _pdf_multivariate_skew_normal(y, vec_xi, mat_diag_omega, mat_omega_bar, vec_alpha):
    """
    Density of Y = \\xi + \\omega Z, where Z \\sim SN_d(0, \\bar{\\Omega}, \\alpha) random vector.
    See Azzalini (2014, Chapter 5), eq (5.3).
    """
    solve = jnp.linalg.solve
    inner = jnp.inner

    dim = jnp.size(y)

    # \Omega = \omega \bar{\Omega} \omega
    mat_omega = mat_diag_omega @ mat_omega_bar @ mat_diag_omega

    # \phi_d(y - \xi ; \Omega)
    phi_d = jscipy.stats.multivariate_normal.pdf(
        y - vec_xi, mean=jnp.repeat(0.0, dim), cov=mat_omega
    )

    # \Phi(  \alpha^{\top} \omega^{-1} (x - \xi) )
    z = solve(mat_omega, y - vec_xi)
    phi = jscipy.stats.norm.cdf(x=inner(vec_alpha, z))

    return 2 * phi_d * phi


def _pdf_affine_transformation_multivariate_skew_normal(
    x, vec_c, mat_A, vec_xi, mat_diag_omega, mat_omega_bar, vec_alpha, nu
):
    """
    Density of X = c + A^{\\top} Y, where Y \\sim ST_d(\\xi, \\Omega, \\alpha, nu). See Azzalini (2014) eq (6.28).
    In particular, the density of X follows ST_h(\\xi_X, \\Omega_X, \\alpha_X, \\nu)

    Params
    ------
    A : Full-rank d x h matrix, h <= d
    c : Vector of size d
    """
    transpose = jnp.transpose
    inv = jnp.linalg.inv
    sqrt = jnp.sqrt
    diag = jnp.diag

    dim = jnp.size(x)

    # \xi_X = c + A^{\top} \xi
    vec_xi_X = vec_c + transpose(mat_A) @ vec_xi

    # \Omega_X = A^{\top} \Omega A
    mat_omega = mat_diag_omega @ mat_omega_bar @ mat_diag_omega
    mat_omega_X = transpose(mat_A) @ mat_omega @ mat_A
    mat_omega_X_inv = inv(mat_omega_X)

    # \omega_X
    mat_diag_omega_X = diag(diag(mat_omega_X))

    # \omega_X^{-1}
    mat_diag_omega_X_inv = diag(1 / diag(mat_diag_omega_X))

    # \bar{\Omega}_X
    mat_omega_X_bar = mat_diag_omega_X_inv @ mat_omega_X @ mat_diag_omega_X_inv

    # \delta
    vec_delta = _calc_skew_delta(vec_alpha=vec_alpha, mat_omega_bar=mat_omega_bar)
    vec_delta = vec_delta.reshape(dim, 1)

    _vec_alpha_X_denominator = sqrt(
        1
        - transpose(vec_delta)
        @ mat_diag_omega
        @ mat_A
        @ mat_omega_X_inv
        @ transpose(mat_A)
        @ mat_diag_omega
        @ vec_delta
    )
    _vec_alpha_X_denominator = _vec_alpha_X_denominator.reshape(1)

    _vec_alpha_X_numerator = (
        mat_diag_omega_X
        @ mat_omega_X_inv
        @ transpose(mat_A)
        @ mat_diag_omega
        @ vec_delta
    )

    # \alpha_X = (1 - \delta^{\top} \omega A \Omega_X^{-1}A^{\top}\omega\delta)^{-1/2} \omega_X \Omega_X^{-1} A^{-1} \omega \delta
    vec_alpha_X = _vec_alpha_X_numerator / _vec_alpha_X_denominator
    vec_alpha_X = vec_alpha_X.reshape(dim)

    # Compute the pdf of ST_h(\\xi_X, \\Omega_X, \\alpha_X, \\nu)
    _pdf = _pdf_multivariate_skew_t(
        x,
        vec_xi=vec_xi_X,
        mat_diag_omega=mat_diag_omega_X,
        mat_omega_bar=mat_omega_X_bar,
        vec_alpha=vec_alpha_X,
        nu=nu,
    )
    return _pdf


def pdf_standardized_transformation_multivariate_skew_normal(x, vec_alpha, nu):
    """
    Density of X = c + A^{\\top} Y, where Y \\sim ST_d(0, \\bar{\\Omega}, \\alpha, \\nu),
    such that c = -\\Sigma_Y^{-1/2} \\mu_Y, A = \\Sigma_Y^{-1/2}, \\bar{\\Omega} = I_d.

    In other words, X will be skew-t multivariate distribution but with mean zero
    and identity variance-covariance.
    """
    dim = jnp.size(x)

    # Compute the mean vector and covariance matrix of Y \sim ST_d(0, \bar{\Omega}, \alpha, \nu),
    # where we fix \bar{\Omega} = I_d
    mat_omega_bar = jnp.identity(dim)

    # Compute \Sigma_Y
    mat_sigma_Y = cov_normalized_multivariate_skew_t(
        mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha, nu=nu
    )

    # Get A = \Sigma_Y^{-1/2}
    mat_A = calc_psd_matrix_invsqrt(mat_sigma_Y)

    # c = -\Sigma_Y^{-1/2} \mu_Y
    vec_mu_Y = mean_normalized_multivariate_skew_t(
        mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha, nu=nu
    )
    vec_c = -1 * mat_A @ vec_mu_Y

    vec_xi = jnp.repeat(0.0, dim)
    mat_diag_omega = jnp.identity(dim)
    pdf_val = _pdf_affine_transformation_multivariate_skew_normal(
        x,
        vec_c=vec_c,
        mat_A=mat_A,
        vec_xi=vec_xi,
        mat_diag_omega=mat_diag_omega,
        mat_omega_bar=mat_omega_bar,
        vec_alpha=vec_alpha,
        nu=nu,
    )
    return pdf_val


def log_likelihood_normalized_multivariate_skew_normal(
    mat_omega_bar, vec_alpha, data: ArrayLike
) -> DTypeLike:
    """
    Negative of log-likelihood of SN_d(0, \\bar{\Omega}, \\alpha) observations.

    Params
    ------
    data: N x d matrix containing the observations.
    """
    _pdf = lambda x: _pdf_normalized_multivariate_skew_normal(
        x, mat_omega_bar, vec_alpha
    )
    _func = vmap(_pdf, in_axes=0)

    summands = _func(data)
    loglik_summands = jnp.log(summands)

    loglik = loglik_summands.mean()
    return -1.0 * loglik


def _sample_normalized_multivariate_skew_normal(
    key: KeyArrayLike, mat_omega_bar: ArrayLike, vec_alpha: ArrayLike
) -> Array:
    """
    Sample a SN_d(0, \bar{\Omega}, \alpha) random vector by using its
    stochastic representation. See Azzalini (2014, Section 5.1.3).
    """
    dim = jnp.size(vec_alpha)

    # Construct block matrix
    # \Omega^{*} =
    # \bar{\Omega}     \delta
    # \delta^{\top}    1
    delta = _calc_skew_delta(vec_alpha=vec_alpha, mat_omega_bar=mat_omega_bar)
    delta = delta.reshape(dim, 1)
    mat_omega_star = jnp.block(
        [[mat_omega_bar, delta], [jnp.transpose(delta), jnp.array([1.0])]]
    )

    # X = [X_0, X_1] \sim N_{d+1}(0, \Omega^{*})
    mean = jnp.repeat(0.0, dim + 1)
    vec_x = jax.random.multivariate_normal(key=key, mean=mean, cov=mat_omega_star)
    vec_x0 = vec_x[:-1]
    x1 = vec_x[-1]

    if x1 > 0:
        return vec_x0
    else:
        return -1 * vec_x0


def _pdf_multivariate_student_t(
    x: ArrayLike, mat_sigma: ArrayLike, nu: DTypeLike
) -> DTypeLike:
    """
    Density of a multivariate Student t's distribution t_d(x ; \Sigma, \nu).
    See Azzalini (2014) eq (6.9).
    """
    pi = jnp.pi
    gamma = jax.scipy.special.gamma
    det = jnp.linalg.det
    inner = jnp.inner
    solve = jnp.linalg.solve

    dim = jnp.array(jnp.size(x))

    _first_term = gamma((nu + dim) / 2) / (
        (nu * pi) ** (dim / 2) * gamma(nu / 2) * det(mat_sigma) ** (1 / 2)
    )

    # Compute x^\top \Sigma^{-1} x
    x_sigmainv_x = inner(x, solve(mat_sigma, x))

    _second_term = (1 + x_sigmainv_x / nu) ** (-(nu + dim) / 2)

    return _first_term * _second_term


def _cdf_t_distribution(x: ArrayLike, nu: float) -> DTypeLike:
    """
    The CDF T(x ; \nu) of a univariate Student-t distribution.

    In particular, if \nu = 1, 2, ..., 5 then we use the exact known
    solution. If \nu > 5, then we use the Li and De Moor (1999) 'A corrected
    normal approximation for the Student's t distribution' approximation.

    Note: More general choices of a positive-valued real number for
    \nu involves the 2F1-hypergeometric function, of which JAX currently
    does not implement (in contrast to scipy).
    """
    pi = jnp.pi
    arctan = jnp.arctan
    sqrt = jnp.sqrt

    if nu == 1:
        return 1 / 2 + (1 / pi) * arctan(x)
    elif nu == 2:
        return 1 / 2 + x / (2 * sqrt(2) * sqrt(1 + x**2 / 2))
    elif nu == 3:
        return 1 / 2 + (1 / pi) * ((x / sqrt(3)) / (1 + x**2 / 3) + arctan(x / sqrt(3)))
    elif nu == 4:
        return 1 / 2 + (3 / 8) * (x / sqrt(1 + x**2 / 4)) * (
            1 - x**2 / (12 * (1 + x**2 / 4))
        )
    elif nu == 5:
        return 1 / 2 + (1 / pi) * (
            x / (sqrt(5) * (1 + x**2 / 5)) * (1 + 2 / (3 * (1 + x**2 / 5)))
            + arctan(x / sqrt(5))
        )
    else:  # Li and De Moor (1999) approximation
        lbda = (4 * nu + x**2 - 1) / (4 * nu + 2 * x**2)
        return jscipy.stats.norm.cdf(lbda * x)


def _pdf_normalized_multivariate_skew_t(
    x: ArrayLike, mat_omega_bar: ArrayLike, vec_alpha: ArrayLike, nu: DTypeLike
):
    """
    Density of a ST_d(0, \bar{\Omega}, \alpha, \nu) random vector.
    See Azzalini (2014, Section 6.2.1)
    """
    inner = jnp.inner
    sqrt = jnp.sqrt
    solve = jnp.linalg.solve

    dim = jnp.size(x)

    # t_d(z; \bar\Omega}, nu) term
    t_d = _pdf_multivariate_student_t(x=x, mat_sigma=mat_omega_bar, nu=nu)

    # T(. ; \nu + d) term
    q_x = inner(x, solve(mat_omega_bar, x))
    z = inner(vec_alpha, x) * sqrt((nu + dim) / (nu + q_x))
    t_dist = _cdf_t_distribution(z, nu + dim)

    return jnp.asarray(2.0) * t_d * t_dist


def _pdf_multivariate_skew_t(y, vec_xi, mat_diag_omega, mat_omega_bar, vec_alpha, nu):
    """
    Density of Y = \\xi + \\omega Z, where Z \\sim ST_d(0, \\bar{\\Omega}, \\alpha) random vector.
    See Azzalini (2014, Chapter 5), eq (5.3).
    """
    solve = jnp.linalg.solve

    # \Omega = \omega \bar{\Omega} \omega
    mat_omega = mat_diag_omega @ mat_omega_bar @ mat_diag_omega

    # z = \omega^{-1}(y - \xi)
    z = solve(mat_omega, y - vec_xi)

    # \det(\omega). Note that we use explicitly the fact that the
    # d x d matrix \omega is diagonal.
    det_omega_diag = jnp.prod(jnp.diag(mat_diag_omega))

    # f_Z(z)
    f_z = _pdf_normalized_multivariate_skew_t(
        z, mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha, nu=nu
    )

    return (1 / det_omega_diag) * f_z


def log_likelihood_normalized_multivariate_skew_t(
    mat_omega_bar, vec_alpha, nu, data: ArrayLike
) -> DTypeLike:
    """
    Negative of log-likelihood of ST_d(0, \bar{\Omega}, \alpha, \nu) observations.

    Params
    ------
    data: N x d matrix containing the observations.
    """
    _func = vmap(_pdf_normalized_multivariate_skew_t, in_axes=[0, None, None, None])

    summands = _func(data, mat_omega_bar, vec_alpha, nu)
    loglik_summands = jnp.log(summands)

    loglik = jnp.sum(loglik_summands)
    return -1.0 * loglik


def _sample_normalized_multivariate_skew_t(
    key: KeyArrayLike, mat_omega_bar: ArrayLike, vec_alpha: ArrayLike, nu: float
) -> Array:
    """
    Sample a ST_d(0, \bar{\Omega}, \alpha, \nu) random vector by using its
    stochastic representation. See Azzalini (2014, Section 6.2.1).

    Note and recall if Z \sim ST_d(0, \bar{\Omega}, \alpha, \nu), then it is
    defined by
    Z = V^{-1/2} Z_0
    where Z_0 \sim SN_d(0, \bar{\Omega}, \alpha) and it is independent of V,
    of which V \sim \chi_\nu^{2} / \nu.
    """
    chisquare = jax.random.chisquare
    sqrt = jnp.sqrt

    # Draw Z_0 \sim SN_d(0, \bar{\Omega}, \alpha)
    z0 = _sample_normalized_multivariate_skew_normal(
        key=key, mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha
    )

    # Draw V \sim \chi_\nu^{2} / \nu that is independent of
    # Z_0
    key, subkey = random.split(key)
    chisq = chisquare(key=key, df=nu)
    v = chisq / nu

    # Construct Skew-t random vector
    z = z0 / sqrt(v)
    return z


def sample_standardized_multivariate_skew_t(
    key: KeyArrayLike, vec_alpha: ArrayLike, nu: float
):
    """
    Draw a ST_d distribution that's standarized with mean zero and
    identity variance.

    Specifically, draw a sample of the random vector X = c + A^{\\top} Y,
    where Y \\sim ST_d(0, \\bar{\\Omega}, \\alpha, \nu), and
    where we set \\bar{\\Omega} = I_d, c = -\\Sigma_Y^{-1/2} \\mu_Y,
    and A = \\Sigma_Y^{-1/2}.
    """

    dim = jnp.size(vec_alpha)

    # Draw Y \sim ST_d(0, \bar{\Omega}, \alpha, \nu)
    mat_omega_bar = jnp.identity(dim)
    vec_Y = _sample_normalized_multivariate_skew_t(
        key=key, mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha, nu=nu
    )

    # Compute the mean and variance of Y
    vec_mu_Y = mean_normalized_multivariate_skew_t(
        mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha, nu=nu
    )
    mat_cov_Y = cov_normalized_multivariate_skew_t(
        mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha, nu=nu
    )

    # Compute A = \Sigma_Y^{-1/2} and c = -\Sigma_Y^{-1/2} \mu_Y
    mat_A = calc_psd_matrix_invsqrt(mat_X=mat_cov_Y)
    vec_c = -1 * mat_A @ vec_mu_Y

    # Compute X = c + A^{\top} Y
    vec_X = vec_c + mat_A @ vec_Y

    return vec_X


def _calc_skew_b(nu: float):
    """
    See Azzalini (2014) eq (4.15).

    Expression
    b_\nu = \sqrt{\nu} \Gamma((\nu - 1) / 2) / ( \sqrt{\pi} \Gamma(\nu / 2))
    """
    sqrt = jnp.sqrt
    gamma = jax.scipy.special.gamma
    pi = jnp.pi

    if nu <= 1:
        raise ValueError("Only vaid for nu > 1")

    return sqrt(nu) * gamma((nu - 1) / 2) / (sqrt(pi) * gamma(nu / 2))


def _calc_skew_delta(vec_alpha, mat_omega_bar) -> float:
    """
    See Azzalini (2014) eq (5.11).

    Expression
    \delta = (1 + \alpha^{\top} \bar{\Omega} \alpha)^{-1/2} \bar{\Omega} \alpha
    """
    _numerator = mat_omega_bar @ vec_alpha  # vector
    _denominator = (1 + jnp.transpose(vec_alpha) @ mat_omega_bar @ vec_alpha) ** (
        -1 / 2
    )

    delta = _numerator / _denominator
    return delta


def mean_normalized_multivariate_skew_normal(mat_omega_bar, vec_alpha):
    """
    First moment of a SN_d(0, \bar{\Omega}, \alpha) random vector.
    """
    b_nu = _calc_skew_b(nu=nu)
    delta = _calc_skew_delta(vec_alpha=vec_alpha, mat_omega_bar=mat_omega_bar)

    mu_z = b_nu * delta
    return mu_z


def cov_normalized_multivariate_skew_normal(mat_omega_bar, vec_alpha):
    """
    Covariance matrix of a SN_d(0, \bar{\Omega}, \alpha) random vector.
    """
    dim = jnp.size(vec_alpha)

    mu_z = mean_normalized_multivariate_skew_normal(
        mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha
    )
    mu_z = mu_z.reshape(dim, 1)

    mat_omega_z = mat_omega_bar - mu_z @ jnp.transpose(mu_z)
    return mat_omega_z


def mean_normalized_multivariate_skew_t(mat_omega_bar, vec_alpha, nu):
    """
    First moment of a ST_d(0, \bar{\Omega}, \alpha, \nu) random vector.
    """
    if nu <= 1:
        raise ValueError("nu <= 1 implies first moment is undefined")

    mu_z = mean_normalized_multivariate_skew_normal(
        mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha
    )
    return mu_z


def cov_normalized_multivariate_skew_t(mat_omega_bar, vec_alpha, nu):
    """
    Covariance matrix of a ST_d(0, \bar{\Omega}, \alpha, \nu) random vector.
    """
    if nu <= 2:
        raise ValueError("nu <= 2 implies second moment is undefined")

    dim = jnp.size(vec_alpha)

    mu_z = mean_normalized_multivariate_skew_normal(
        mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha
    )
    mu_z = mu_z.reshape(dim, 1)

    mat_sigma = nu / (nu - 2) * mat_omega_bar - mu_z @ jnp.transpose(mu_z)
    return mat_sigma


key = random.key(12345)

dim = 2
nu = 5
num_sample = int(1e2)
vec_alpha_true = (1 / 2) * jax.random.uniform(key, shape=(dim,)) - (1 / 4)
# vec_alpha_true = jnp.repeat(0.0, dim)
mat_omega_bar = jnp.identity(dim)


lst_data = []
for i in range(num_sample):
    key, subkey = random.split(key)
    _ = _sample_normalized_multivariate_skew_t(key, mat_omega_bar, vec_alpha_true, nu)
    lst_data.append(_)

data = jnp.array(lst_data)

key, subkey = random.split(key)
vec_c = jnp.repeat(0.0, dim)
_ = 2 * jax.random.uniform(key, shape=(dim, dim)) - 1
mat_A = 0.5 * (_ + jnp.transpose(_))
key, subkey = random.split(key)
vec_xi = 2 * jax.random.uniform(key, shape=(dim,)) - 1
mat_diag_omega = jnp.identity(dim)
mat_omega_bar = jnp.identity(dim)

sample_standardized_multivariate_skew_t(key=key, vec_alpha=vec_alpha_true, nu=nu)

pdf_standardized_transformation_multivariate_skew_normal(
    x=data[0, :], vec_alpha=vec_alpha_true, nu=nu
)

_pdf_affine_transformation_multivariate_skew_normal(
    x=data[0, :],
    vec_c=vec_c,
    mat_A=mat_A,
    vec_xi=vec_xi,
    mat_diag_omega=mat_diag_omega,
    mat_omega_bar=mat_omega_bar,
    vec_alpha=vec_alpha_true,
    nu=nu,
)

# learning_rate = 1e-2
# optimizer = optax.adam(learning_rate)


compute_loss = lambda vec_alpha: log_likelihood_normalized_multivariate_skew_t(
    mat_omega_bar, vec_alpha, nu, data
)

# bears = jax.scipy.optimize.minimize(
#     compute_loss, x0=jnp.array([0.2, 0.2]), method="BFGS"
# )


start_learning_rate = 1e-1
transition_steps = 1000
decay_rate = 0.99
scheduler = optax.exponential_decay(
    init_value=start_learning_rate,
    transition_steps=transition_steps,
    decay_rate=decay_rate,
)
gradient_transform = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.scale_by_schedule(scheduler),
    optax.scale(-1.0),
)

key, subkey = random.split(key)
vec_alpha = 2 * jax.random.uniform(key, shape=(dim,)) - 1
opt_state = gradient_transform.init(vec_alpha)

num_loops = 100
score = jax.grad(compute_loss)
for _ in range(num_loops):
    grads = score(vec_alpha)
    updates, opt_state = gradient_transform.update(grads, opt_state)
    vec_alpha = optax.apply_updates(vec_alpha, updates)


first_moment = jnp.mean(data, axis=0)

mean_true = mean_normalized_multivariate_skew_t(
    mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha_true, nu=nu
)

mean_est = mean_normalized_multivariate_skew_t(
    mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha, nu=nu
)

cov_true = cov_normalized_multivariate_skew_t(
    mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha_true, nu=nu
)

cov_est = cov_normalized_multivariate_skew_t(
    mat_omega_bar=mat_omega_bar, vec_alpha=vec_alpha, nu=nu
)

sample_cov = jnp.cov(data, rowvar=False)

jnp.linalg.norm(vec_alpha_true - vec_alpha) / jnp.size(vec_alpha_true)

jnp.linalg.norm(mean_true - mean_est)
jnp.linalg.norm(mean_true - first_moment)

jnp.linalg.norm(cov_true - cov_est)
jnp.linalg.norm(cov_true - sample_cov)

breakpoint()
