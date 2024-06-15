from enum import verify
from threading import excepthook
import jax
from jax._src.random import KeyArrayLike
import jax.numpy as jnp
import jax.scipy as jscipy
import jax.scipy.optimize
from jax import grad, jit, vmap
from jax import random
import jax.test_util
from jax import Array
from jax.typing import ArrayLike, DTypeLike
import jax.typing as jpt
import numpy.typing as npt

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="dcc.log",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(message)s",
    filemode="w",
)


import typing as tp


import itertools
import functools
from functools import partial

import optax
import optimistix
import jaxopt

import numpy as np
from numpy._typing import NDArray
import scipy
import matplotlib.pyplot as plt

import sgt

# HACK:
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod
# jax.config.update("jax_debug_nans", True)  # Should disable in full prod


logger = logging.getLogger(__name__)

DICT_INIT_T0_CONDITIONS = "dict_init_t0_conditions"
DICT_PARAMS_MEAN = "dict_params_mean"
DICT_PARAMS_Z = "dict_params_z"
DICT_PARAMS_DCC_UVAR_VOL = "dict_params_dcc_uvar_vol"
DICT_PARAMS_DCC_MVAR_COR = "dict_params_dcc_mvar_cor"

# Time t = 0 initial conditions
VEC_SIGMA_0 = "vec_sigma_0"
MAT_Q_0 = "mat_Q_0"

# SGT parameters
VEC_LBDA = "vec_lbda"
VEC_P0 = "vec_p0"
VEC_Q0 = "vec_q0"

# Mean return parameters
VEC_MU = "vec_mu"

# Univariate volatilities parameters
VEC_OMEGA = "vec_omega"
VEC_BETA = "vec_beta"
VEC_ALPHA = "vec_alpha"
VEC_PSI = "vec_psi"

# Multivariate DCC parameters
VEC_DELTA = "vec_delta"
MAT_QBAR = "mat_Qbar"


def _generate_random_cov_mat(key: KeyArrayLike, dim: int) -> Array:
    """
    Generate a random covariance-variance matrix (i.e. symmetric and PSD).
    """
    mat_A = jax.random.normal(key, shape=(dim, dim)) * 0.25
    mat_sigma = jnp.transpose(mat_A) @ mat_A

    return mat_sigma


def _calc_demean_returns(mat_returns: Array, vec_mu: NDArray) -> Array:
    """
    Calculate \\epsilon_t = R_t - \\mu_t
    """
    return mat_returns - vec_mu


@jax.jit
def _calc_unexpected_excess_rtn(mat_Sigma: Array, vec_z: Array) -> Array:
    """
    Return \\epsilon_t = \\Sigma_t^{1/2} z_t.

    Given a covariance matrix \\Sigma and an innovation
    vector z, return the unexpected excess returns vector
    \\epsilon = \\Sigma^{1/2} z, where \\Sigma^{1/2} is
    taken to be the Cholesky decomposition of \\Sigma.
    """
    mat_Sigma_sqrt = jscipy.linalg.cholesky(mat_Sigma)
    vec_epsilon = mat_Sigma_sqrt @ vec_z

    return vec_epsilon


@jax.jit
def _calc_normalized_unexpected_excess_rtn(
    vec_sigma: Array, vec_epsilon: Array
) -> Array:
    """
    Return u_t = D_t^{-1} \\epsilon_t
    """
    mat_inv_D = jnp.diag(1 / vec_sigma)
    vec_u = mat_inv_D @ vec_epsilon

    return vec_u


@jax.jit
def _calc_mat_Q(
    vec_delta: NDArray,
    vec_u_t_minus_1: Array,
    mat_Q_t_minus_1: Array,
    mat_Qbar: Array,
) -> Array:
    """
    Return the Q_t matrix of the DCC model
    """
    mat_Q_t = (
        (1 - vec_delta[0] - vec_delta[1]) * mat_Qbar
        + vec_delta[0] * (jnp.outer(vec_u_t_minus_1, vec_u_t_minus_1))
        + vec_delta[1] * mat_Q_t_minus_1
    )
    return mat_Q_t


@jax.jit
def _calc_mat_Gamma(mat_Q: Array) -> Array:
    """
    Return the \\Gamma_t matrix of the DCC model
    """
    mat_Qstar_inv_sqrt = jnp.diag(jnp.diag(mat_Q) ** (-1 / 2))
    mat_Gamma = mat_Qstar_inv_sqrt @ mat_Q @ mat_Qstar_inv_sqrt

    return mat_Gamma


@jax.jit
def _calc_mat_Sigma(vec_sigma: Array, mat_Gamma: Array) -> Array:
    """
    Return the covariance \\Sigma_t = D_t \\Gamma_t D_t,
    where D_t is a diagonal matrix of \\sigma_{i,t}.
    """
    mat_D = jnp.diag(vec_sigma)

    mat_Sigma = mat_D @ mat_Gamma @ mat_D
    return mat_Sigma


def simulate_dcc(
    key: KeyArrayLike,
    data_z: Array,
    vec_omega: NDArray,
    vec_beta: NDArray,
    vec_alpha: NDArray,
    vec_psi: NDArray,
    vec_delta: NDArray,
    mat_Qbar: Array,
) -> tp.Tuple[Array, Array]:
    """
    Simulate a DCC model
    """
    logger.info("Begin DCC simulation.")

    num_sample = data_z.shape[0]
    dim = data_z.shape[1]

    # Initial conditions at t = 0
    vec_z_0 = data_z[0, :]
    mat_Sigma_0 = _generate_random_cov_mat(key=key, dim=dim)
    vec_sigma_0 = jnp.sqrt(jnp.diag(mat_Sigma_0))
    vec_epsilon_0 = _calc_unexpected_excess_rtn(mat_Sigma=mat_Sigma_0, vec_z=vec_z_0)
    vec_u_0 = _calc_normalized_unexpected_excess_rtn(
        vec_sigma=vec_sigma_0, vec_epsilon=vec_epsilon_0
    )
    key, _ = random.split(key)
    mat_Q_0 = _generate_random_cov_mat(key=key, dim=dim)

    # Init
    lst_epsilon = [jnp.empty(dim)] * num_sample
    lst_sigma = [jnp.empty(dim)] * num_sample
    lst_u = [jnp.empty(dim)] * num_sample
    lst_Q = [jnp.empty((dim, dim))] * num_sample
    lst_Sigma = [jnp.empty((dim, dim))] * num_sample

    # Save initial conditions
    lst_epsilon[0] = vec_epsilon_0
    lst_sigma[0] = vec_sigma_0
    lst_u[0] = vec_u_0
    lst_Q[0] = mat_Q_0
    lst_Sigma[0] = mat_Sigma_0

    # Iterate
    for tt in range(1, num_sample):
        # Set t - 1 quantities
        vec_epsilon_t_minus_1 = lst_epsilon[tt - 1]
        vec_sigma_t_minus_1 = lst_sigma[tt - 1]
        vec_u_t_minus_1 = lst_u[tt - 1]
        mat_Q_t_minus_1 = lst_Q[tt - 1]

        # Compute \\sigma_{i,t}^2
        vec_sigma2_t = _calc_asymmetric_garch_sigma2(
            vec_sigma_t_minus_1=vec_sigma_t_minus_1,
            vec_epsilon_t_minus_1=vec_epsilon_t_minus_1,
            vec_omega=vec_omega,
            vec_beta=vec_beta,
            vec_alpha=vec_alpha,
            vec_psi=vec_psi,
        )
        vec_sigma_t = jnp.sqrt(vec_sigma2_t)

        # Compute Q_t
        mat_Q_t = _calc_mat_Q(
            vec_delta=vec_delta,
            vec_u_t_minus_1=vec_u_t_minus_1,
            mat_Q_t_minus_1=mat_Q_t_minus_1,
            mat_Qbar=mat_Qbar,
        )

        # Compute Gamma_t
        mat_Gamma_t = _calc_mat_Gamma(mat_Q=mat_Q_t)

        # Compute \Sigma_t
        mat_Sigma_t = _calc_mat_Sigma(vec_sigma=vec_sigma_t, mat_Gamma=mat_Gamma_t)

        # Compute \epsilon_t
        vec_z = data_z[tt, :]
        vec_epsilon_t = _calc_unexpected_excess_rtn(mat_Sigma=mat_Sigma_t, vec_z=vec_z)

        # Compute u_t
        vec_u_t = _calc_normalized_unexpected_excess_rtn(
            vec_sigma=vec_sigma_t, vec_epsilon=vec_epsilon_t
        )

        # Bookkeeping
        lst_epsilon[tt] = vec_epsilon_t
        lst_sigma[tt] = vec_sigma_t
        lst_u[tt] = vec_u_t
        lst_Q[tt] = mat_Q_t
        lst_Sigma[tt] = mat_Sigma_t

    # Convenient output form
    mat_epsilon = jnp.array(lst_epsilon)
    tns_Sigma = jnp.array(lst_Sigma)
    return mat_epsilon, tns_Sigma


@jax.jit
def _calc_asymmetric_garch_sigma2(
    vec_sigma_t_minus_1, vec_epsilon_t_minus_1, vec_omega, vec_beta, vec_alpha, vec_psi
):
    """
    Compute \\sigma_{i,t}^2 of the asymmetric-GARCH(1,1) model
    """
    vec_sigma2_t = (
        vec_omega
        + vec_beta * vec_sigma_t_minus_1**2
        + vec_alpha * vec_epsilon_t_minus_1**2
        + vec_psi * vec_epsilon_t_minus_1**2 * (vec_epsilon_t_minus_1 < 0)
    )
    return vec_sigma2_t


def _calc_trajectory_uvar_vol(
    mat_epsilon: Array,
    vec_sigma_0: Array,
    vec_omega: NDArray,
    vec_beta: NDArray,
    vec_alpha: NDArray,
    vec_psi: NDArray,
) -> Array:
    """
    Calculate the trajectory of univariate vol's {\\sigma_{i,t}}
    for all t based on the asymmetric-GARCH(1,1) model
    """
    num_sample = mat_epsilon.shape[0]
    dim = mat_epsilon.shape[1]

    mat_sigma = jnp.empty(shape=(num_sample, dim))
    mat_sigma = mat_sigma.at[0].set(vec_sigma_0)

    def _body_fun(tt, mat_sigma):
        vec_sigma_t_minus_1 = mat_sigma[tt - 1, :]
        vec_epsilon_t_minus_1 = mat_epsilon[tt - 1, :]

        vec_sigma2_t = _calc_asymmetric_garch_sigma2(
            vec_sigma_t_minus_1=vec_sigma_t_minus_1,
            vec_epsilon_t_minus_1=vec_epsilon_t_minus_1,
            vec_omega=vec_omega,
            vec_beta=vec_beta,
            vec_alpha=vec_alpha,
            vec_psi=vec_psi,
        )
        vec_sigma_t = jnp.sqrt(vec_sigma2_t)

        mat_sigma = mat_sigma.at[tt].set(vec_sigma_t)
        return mat_sigma

    mat_sigma = jax.lax.fori_loop(
        lower=1, upper=num_sample, body_fun=_body_fun, init_val=mat_sigma
    )

    return mat_sigma


def _calc_trajectory_mvar_cov(
    mat_epsilon: Array,
    mat_sigma: Array,
    mat_Q_0: Array,
    vec_delta: NDArray,
    mat_Qbar: Array,
) -> Array:
    """
    Calculate the trajectory of multivariate covariances
    {\\Sigma_t} based on the DCC model.
    """
    num_sample = mat_epsilon.shape[0]
    dim = mat_epsilon.shape[1]

    mat_u = jnp.empty(shape=(num_sample, dim))
    tns_Q = jnp.empty(shape=(num_sample, dim, dim))
    tns_Sigma = jnp.empty(shape=(num_sample, dim, dim))

    vec_u_0 = _calc_normalized_unexpected_excess_rtn(
        vec_sigma=mat_sigma[0, :], vec_epsilon=mat_epsilon[0, :]
    )
    mat_Gamma_0 = _calc_mat_Gamma(mat_Q=mat_Q_0)
    mat_Sigma_0 = _calc_mat_Sigma(vec_sigma=mat_sigma[0, :], mat_Gamma=mat_Gamma_0)

    mat_u = mat_u.at[0].set(vec_u_0)
    tns_Q = tns_Q.at[0].set(mat_Q_0)
    tns_Sigma = tns_Sigma.at[0].set(mat_Sigma_0)

    def _body_fun(tt, carry):
        mat_u, tns_Q, tns_Sigma = carry

        # Compute Q_t
        mat_Q_t = _calc_mat_Q(
            vec_delta=vec_delta,
            vec_u_t_minus_1=mat_u[tt - 1, :],
            mat_Q_t_minus_1=tns_Q[tt - 1, :, :],
            mat_Qbar=mat_Qbar,
        )

        # Compute Gamma_t
        mat_Gamma_t = _calc_mat_Gamma(mat_Q=mat_Q_t)

        # Compute \Sigma_t
        mat_Sigma_t = _calc_mat_Sigma(vec_sigma=mat_sigma[tt, :], mat_Gamma=mat_Gamma_t)

        # Compute u_t
        vec_u_t = _calc_normalized_unexpected_excess_rtn(
            vec_sigma=mat_sigma[tt, :], vec_epsilon=mat_epsilon[tt, :]
        )

        # Bookkeeping
        mat_u = mat_u.at[tt].set(vec_u_t)
        tns_Q = tns_Q.at[tt].set(mat_Q_t)
        tns_Sigma = tns_Sigma.at[tt].set(mat_Sigma_t)

        return mat_u, tns_Q, tns_Sigma

    carry = jax.lax.fori_loop(
        lower=1,
        upper=num_sample,
        body_fun=_body_fun,
        init_val=(mat_u, tns_Q, tns_Sigma),
    )
    _, _, tns_Sigma = carry

    return tns_Sigma


def _calc_trajectory_normalized_unexp_returns(
    mat_sigma: Array, mat_epsilon: Array
) -> Array:
    """
    Calculate the trajectory of u_t = D_t^{-1}\\epsilon_t
    """
    mat_u = mat_epsilon / mat_sigma
    return mat_u


def simulate_returns(
    seed: int,
    dim: int,
    num_sample: int,
    dict_params_mean,
    dict_params_z,
    dict_params_dcc_uvar_vol,
    dict_params_dcc_mvar_cor,
    num_cores: int,
):
    key = random.key(seed)

    # Simulate the innovations
    data_z = sgt.sample_mvar_sgt(
        key=key, num_sample=num_sample, num_cores=num_cores, **dict_params_z
    )

    # Simulate a DCC model and obtain \epsilon_t
    mat_epsilon, _ = simulate_dcc(
        key=key, data_z=data_z, **dict_params_dcc_uvar_vol, **dict_params_dcc_mvar_cor
    )

    # Set the asset mean
    vec_mu = dict_params_mean["vec_mu"]

    # Asset returns
    mat_returns = vec_mu + mat_epsilon

    # Sanity checks
    if mat_returns.shape != (num_sample, dim):
        logger.error("Incorrect shape for the simulated returns.")

    logger.info("Done simulating returns")
    return data_z, mat_returns


def _calc_innovations(vec_epsilon: Array, mat_Sigma: Array) -> Array:
    """
    Return innovations z_t = \\Sigma_t^{-1/2} \\epsilon_t, where we are
    given \\epsilon_t = R_t - \\mu_t and conditional covariances
    {\\Sigma_t}
    """
    mat_Sigma_sqrt = jnp.linalg.cholesky(mat_Sigma, upper=True)
    vec_z = jnp.linalg.solve(mat_Sigma_sqrt, vec_epsilon)

    return vec_z


def _calc_trajectory_innovations(mat_epsilon: Array, tns_Sigma: Array) -> Array:
    """
    Calculate trajectory of innovations z_t's over the full sample.
    """
    _func = vmap(_calc_innovations, in_axes=[0, 0])
    mat_z = _func(mat_epsilon, tns_Sigma)

    return mat_z


def _calc_trajectories(
    mat_returns: Array,
    dict_init_t0_conditions: dict[str, NDArray | Array],
    dict_params_mean: dict[str, NDArray],
    dict_params_dcc_uvar_vol: dict[str, NDArray | Array],
    dict_params_dcc_mvar_cor: dict[str, NDArray | Array],
) -> tuple[Array, Array, Array, Array, Array]:
    """
    Given parameters, return the trajectories {\\epsilon_t}, {\\sigma_{i,t}},
    {\\Sigma_t}, {z_t}, {u_t}
    """
    vec_sigma_0 = dict_init_t0_conditions[VEC_SIGMA_0]
    mat_Q_0 = dict_init_t0_conditions[MAT_Q_0]

    # Compute \epsilon_t = R_t - \mu_t
    mat_epsilon = _calc_demean_returns(mat_returns=mat_returns, **dict_params_mean)

    # Calculate the univariate vol's \sigma_{i,t}'s
    mat_sigma = _calc_trajectory_uvar_vol(
        mat_epsilon=mat_epsilon, vec_sigma_0=vec_sigma_0, **dict_params_dcc_uvar_vol
    )

    # Calculate the multivariate covariance \Sigma_t
    tns_Sigma = _calc_trajectory_mvar_cov(
        mat_epsilon=mat_epsilon,
        mat_sigma=mat_sigma,
        mat_Q_0=mat_Q_0,
        **dict_params_dcc_mvar_cor,
    )

    # Calculate the innovations z_t = \Sigma_t^{-1/2} \epsilon_t
    mat_z = _calc_trajectory_innovations(mat_epsilon=mat_epsilon, tns_Sigma=tns_Sigma)

    # Calculate the normalized unexpected returns u_t
    mat_u = _calc_trajectory_normalized_unexp_returns(
        mat_sigma=mat_sigma, mat_epsilon=mat_epsilon
    )

    return mat_epsilon, mat_sigma, tns_Sigma, mat_z, mat_u


def dcc_loglik(
    mat_returns,
    dict_init_t0_conditions,
    dict_params_mean,
    dict_params_z,
    dict_params_dcc_uvar_vol,
    dict_params_dcc_mvar_cor,
    neg_loglik: bool = True,
) -> DTypeLike:
    """
    (Negative) of the likelihood of the DCC-Asymmetric GARCH(1,1) model
    """
    num_sample = mat_returns.shape[0]

    _, _, tns_Sigma, mat_z, _ = _calc_trajectories(
        mat_returns=mat_returns,
        dict_init_t0_conditions=dict_init_t0_conditions,
        dict_params_mean=dict_params_mean,
        dict_params_dcc_uvar_vol=dict_params_dcc_uvar_vol,
        dict_params_dcc_mvar_cor=dict_params_dcc_mvar_cor,
    )

    # Compute {\log\det \Sigma_t}
    _, vec_logdet_Sigma = jnp.linalg.slogdet(tns_Sigma)

    # Compute the log-likelihood of g(z_t) where g \sim SGT
    sgt_loglik = sgt.loglik_mvar_indp_sgt(data=mat_z, neg_loglik=False, **dict_params_z)

    # Objective function of DCC model
    loglik = sgt_loglik - 0.5 * vec_logdet_Sigma.sum()

    if neg_loglik:
        loglik = -1 * loglik

    return loglik


def _make_dict_params_z(x, dim) -> tp.Dict[str, NDArray]:
    """
    Take a vector x and split them into parameters related to the
    z_t \\sim SGT distribution
    """
    try:
        if jnp.size(x) != 3 * dim:
            raise ValueError(
                "Total number of parameters for the SGT process is incorrect."
            )
    except Exception as e:
        logger.error(str(e))

    dict_params_z = {
        VEC_LBDA: x[0:dim],
        VEC_P0: x[dim : 2 * dim],
        VEC_Q0: x[2 * dim :],
    }

    return dict_params_z


def _make_dict_params_mean(x, dim) -> tp.Dict[str, NDArray]:
    """
    Take a vector x and split them into parameters related to the
    mean \\mu
    """
    try:
        if jnp.size(x) != dim:
            raise ValueError(
                "Total number of parameters for the constant mean process is incorrect."
            )
    except Exception as e:
        logger.error(str(e))

    dict_params_mean = {VEC_MU: x}

    return dict_params_mean


def _make_dict_params_dcc_uvar_vol(x, dim) -> tp.Dict[str, NDArray]:
    """
    Take a vector x and split them into parameters related to the
    univariate GARCH processes.
    """
    dict_params_dcc_uvar_vol = {
        VEC_OMEGA: x[0:dim],
        VEC_BETA: x[dim : 2 * dim],
        VEC_ALPHA: x[2 * dim : 3 * dim],
        VEC_PSI: x[3 * dim :],
    }

    return dict_params_dcc_uvar_vol


def _make_dict_params_dcc_mvar_cor(
    x,
    dim,
    mat_returns,
    dict_init_t0_conditions,
    dict_params_mean,
    dict_params_z,  # Unused
    dict_params_dcc_uvar_vol,
    dict_params_dcc_mvar_cor,
) -> tp.Dict[str, NDArray]:
    """
    Take a vector x and split them into parameters related to the
    DCC process.
    """
    try:
        if jnp.size(x) != 2:
            raise ValueError(
                "Total number of parameters for the DCC process is incorrect."
            )
    except Exception as e:
        logger.error(str(e))

    dict_params_dcc_mvar_cor[VEC_DELTA] = x

    # Special treatment estimating \bar{Q}. Apply the ``variance-
    # targetting'' approach. That is, we estimate by
    # \hat{\bar{Q}} = sample moment of \hat{u}_t\hat{u}_t^{\top}.
    _, _, _, _, mat_u = _calc_trajectories(
        mat_returns=mat_returns,
        dict_init_t0_conditions=dict_init_t0_conditions,
        dict_params_mean=dict_params_mean,
        dict_params_dcc_uvar_vol=dict_params_dcc_uvar_vol,
        dict_params_dcc_mvar_cor=dict_params_dcc_mvar_cor,
    )

    # Set \hat{\bar{Q}} = \frac{1}{T} \sum_t u_tu_t^{\top}
    _func = vmap(jnp.outer, in_axes=[0, 0])
    tens_uuT = _func(mat_u, mat_u)
    mat_Qbar = tens_uuT.mean(axis=0)
    dict_params_dcc_mvar_cor[MAT_QBAR] = mat_Qbar

    return dict_params_dcc_mvar_cor


# def _check_params_z(
#     dim: int, dict_params_z: tp.Dict[str, NDArray | Array], min_moments: int = 4
# ) -> bool:
#     """
#     Check the valididty of parameters of z_t.
#     """
#     # HACK:
#     # vec_lbda = dict_params_z[VEC_LBDA]
#     # vec_p0 = dict_params_z[VEC_P0]
#     # vec_q0 = dict_params_z[VEC_Q0]
#     #
#     # if jnp.size(vec_lbda) != dim:
#     #     return False
#     #
#     # if jnp.size(vec_p0) != dim:
#     #     return False
#     #
#     # if jnp.size(vec_q0) != dim:
#     #     return False
#     #
#     # if jnp.any(vec_lbda <= -1):
#     #     return False
#     #
#     # if jnp.any(vec_lbda >= -1):
#     #     return False
#     #
#     # if jnp.any(vec_p0 <= 0):
#     #     return False
#     #
#     # if jnp.any(vec_q0 <= 0):
#     #     return False
#     #
#     # if jnp.any(vec_p0 * vec_q0 < min_moments):
#     #     return False
#
#     return True
#
#
# def _check_params_dcc_uvar_vol(
#     dim: int, dict_params_dcc_uvar_vol: tp.Dict[str, NDArray | Array]
# ) -> bool:
#     """
#     Check the valididty of parameters of univariate volatilities.
#     """
#     # HACK:
#     # for _, val in dict_params_dcc_uvar_vol.items():
#     #     if jnp.size(val) != dim:
#     #         return False
#     #
#     #     if jnp.any(val) < 0:
#     #         return False
#
#     return True
#
#
# def _check_params_dcc_mvar_cor(
#     dim: int, dict_params_dcc_mvar_cor: tp.Dict[str, NDArray | Array]
# ) -> bool:
#     """
#     Check the valididty of DCC parameters
#     """
#     # HACK:
#     # vec_delta = dict_params_dcc_mvar_cor[VEC_DELTA]
#     # mat_Qbar = dict_params_dcc_mvar_cor[MAT_QBAR]
#     #
#     # # Check constraints on \delta_1, \delta_2
#     # if jnp.size(vec_delta) != 2:
#     #     return False
#     #
#     # if jnp.any(vec_delta < 0):
#     #     return False
#     #
#     # if jnp.any(vec_delta > 1):
#     #     return False
#     #
#     # if jnp.sum(vec_delta > 1):
#     #     return False
#     #
#     # # Check constraints on \bar{Q}
#     # # \bar{Q} should be PSD. But for speed purposes, just
#     # # check dimensions.
#     # if mat_Qbar.shape != (dim, dim):
#     #     return False
#
#     return True


def _objfun_dcc_loglik(
    x,
    make_dict_params_fun,
    dict_params: tp.Dict[str, tp.Dict[str, Array]] | tp.Dict[str, tp.Dict[str, Array]],
    dict_init_t0_conditions: tp.Dict[str, Array] | tp.Dict[str, NDArray],
    mat_returns: Array,
) -> DTypeLike | float:
    dim = mat_returns.shape[1]

    # Construct the dict for the parameters
    # that are to be optimized over
    optimizing_params_name = make_dict_params_fun.__name__
    if optimizing_params_name == "_make_dict_params_dcc_mvar_cor":
        # Special treatment in handling the estimate of
        # \hat{\bar{Q}} (i.e. volatility-targetting estimation method)
        dict_optimizing_params = make_dict_params_fun(
            x=x,
            dim=dim,
            mat_returns=mat_returns,
            dict_init_t0_conditions=dict_init_t0_conditions,
            **dict_params,
        )
    else:
        dict_optimizing_params = make_dict_params_fun(x=x, dim=dim)

    # Prettier name
    optimizing_params_name = optimizing_params_name.split("_make_")[1]

    # Update with the rest of the parameters
    # that are held fixed
    dict_params[optimizing_params_name] = dict_optimizing_params

    try:
        neg_loglik = dcc_loglik(
            mat_returns=mat_returns,
            dict_init_t0_conditions=dict_init_t0_conditions,
            neg_loglik=True,
            **dict_params,
        )
    except FloatingPointError:
        logger.info(f"Invalid optimizing parameters.")
        logger.info(f"{dict_params}")
        neg_loglik = jnp.inf

    except Exception as e:
        logger.info(f"{e}")
        neg_loglik = jnp.inf

    return neg_loglik


def _make_params_array_from_dict_params(
    dict_params: tp.Dict[str, jax.Array] | tp.Dict[str, npt.NDArray]
) -> Array:
    """
    Take in a dictionary of parameters and flatten them to an
    array in preparation for optimization.
    """
    x0 = itertools.chain.from_iterable(dict_params.values())
    x0 = list(x0)
    x0 = np.array(x0)
    x0 = jnp.array(x0)

    return x0


def dcc_garch_optimization(
    mat_returns: Array,
    dict_params_init_guess: tp.Dict[str, Array] | tp.Dict[str, NDArray],
    dict_init_t0_conditions: tp.Dict[str, Array] | tp.Dict[str, NDArray],
    verbose: bool = False,
    grand_maxiter: int = 5,
    maxiter: int = 100,
    tol: float = 1e-3,
    solver_z=jaxopt.LBFGS,
    solver_mean=jaxopt.LBFGS,
    solver_dcc_uvar_vol=jaxopt.LBFGS,
    solver_dcc_mvar_cor=jaxopt.LBFGS,
):
    """
    Run DCC-GARCH optimization
    """
    logger.info(f"Begin DCC-GARCH optimization.")

    #################################################################
    ## Setup partial objective functions
    #################################################################
    objfun_dcc_loglik_opt_params_z = lambda x, dict_params: _objfun_dcc_loglik(
        x=x,
        dict_params=dict_params,
        make_dict_params_fun=_make_dict_params_z,
        dict_init_t0_conditions=dict_init_t0_conditions,
        mat_returns=mat_returns,
    )

    objfun_dcc_loglik_opt_params_mean = lambda x, dict_params: _objfun_dcc_loglik(
        x=x,
        dict_params=dict_params,
        make_dict_params_fun=_make_dict_params_mean,
        dict_init_t0_conditions=dict_init_t0_conditions,
        mat_returns=mat_returns,
    )

    objfun_dcc_loglik_opt_params_dcc_uvar_vol = (
        lambda x, dict_params: _objfun_dcc_loglik(
            x=x,
            dict_params=dict_params,
            make_dict_params_fun=_make_dict_params_dcc_uvar_vol,
            dict_init_t0_conditions=dict_init_t0_conditions,
            mat_returns=mat_returns,
        )
    )

    objfun_dcc_loglik_opt_params_dcc_mvar_cor = (
        lambda x, dict_params: _objfun_dcc_loglik(
            x=x,
            dict_params=dict_params,
            make_dict_params_fun=_make_dict_params_dcc_mvar_cor,
            dict_init_t0_conditions=dict_init_t0_conditions,
            mat_returns=mat_returns,
        )
    )

    neg_loglik_optval = None
    dict_params = dict_params_init_guess
    for iter in range(grand_maxiter):
        #################################################################
        ## Step 1: Optimize for the parameters of z_t
        #################################################################
        x0_z = _make_params_array_from_dict_params(dict_params[DICT_PARAMS_Z])
        solver_obj_z = solver_z(
            fun=objfun_dcc_loglik_opt_params_z,
            verbose=verbose,
            maxiter=maxiter,
            tol=tol,
        )
        sol_z = solver_obj_z.run(init_params=x0_z, dict_params=dict_params)
        dict_params_z_est = _make_dict_params_z(x=sol_z.params, dim=dim)
        dict_params[DICT_PARAMS_Z] = dict_params_z_est

        #################################################################
        ## Step 2: Optimize for the parameters of the mean \mu
        #################################################################
        x0_mean = _make_params_array_from_dict_params(dict_params[DICT_PARAMS_MEAN])
        solver_obj_mean = solver_mean(
            fun=objfun_dcc_loglik_opt_params_mean,
            verbose=verbose,
            maxiter=maxiter,
            tol=tol,
        )
        sol_mean = solver_obj_mean.run(init_params=x0_mean, dict_params=dict_params)
        dict_params_mean_est = _make_dict_params_mean(x=sol_mean.params, dim=dim)
        dict_params[DICT_PARAMS_MEAN] = dict_params_mean_est

        #################################################################
        ## Step 3: Optimize for the parameters of the univariate vol's
        ##         \sigma_{i,t}'s
        #################################################################
        x0_dcc_uvar_vol = _make_params_array_from_dict_params(
            dict_params[DICT_PARAMS_DCC_UVAR_VOL]
        )
        solver_obj_dcc_uvar_vol = solver_dcc_uvar_vol(
            fun=objfun_dcc_loglik_opt_params_dcc_uvar_vol,
            verbose=verbose,
            maxiter=maxiter,
            tol=tol,
        )
        sol_dcc_uvar_vol = solver_obj_dcc_uvar_vol.run(
            init_params=x0_dcc_uvar_vol, dict_params=dict_params
        )
        dict_params_dcc_uvar_vol_est = _make_dict_params_dcc_uvar_vol(
            x=sol_dcc_uvar_vol.params, dim=dim
        )
        dict_params[DICT_PARAMS_DCC_UVAR_VOL] = dict_params_dcc_uvar_vol_est

        #################################################################
        ## Step 4: Optimize for the parameters of the multivariate DCC
        ##         Q_t's
        #################################################################
        # Note special treatment for the initial guess for the
        # DCC multivariate parameters
        x0_dcc_mvar_cor = dict_params[DICT_PARAMS_DCC_MVAR_COR][VEC_DELTA]
        solver_obj_dcc_mvar_cor = solver_dcc_mvar_cor(
            fun=objfun_dcc_loglik_opt_params_dcc_mvar_cor,
            verbose=verbose,
            maxiter=maxiter,
            tol=tol,
        )
        sol_dcc_mvar_cor = solver_obj_dcc_mvar_cor.run(
            init_params=x0_dcc_mvar_cor, dict_params=dict_params
        )
        dict_params_dcc_mvar_cor_est = _make_dict_params_dcc_mvar_cor(
            x=sol_dcc_mvar_cor.params,
            dim=dim,
            mat_returns=mat_returns,
            dict_init_t0_conditions=dict_init_t0_conditions,
            **dict_params,
        )
        dict_params[DICT_PARAMS_DCC_MVAR_COR] = dict_params_dcc_mvar_cor_est

        # Record the value of the last negative log-likelihood
        if iter == grand_maxiter - 1:
            neg_loglik_optval = sol_dcc_mvar_cor.state.value

        logger.info(f"... done iteration {iter}/{grand_maxiter}")

    logger.info(f"Done DCC-GARCH optimization.")
    return neg_loglik_optval, dict_params


if __name__ == "__main__":
    seed = 1234567
    key = random.key(seed)
    rng = np.random.default_rng(seed)
    num_sample = int(3e3)
    dim = 3
    num_cores = 8

    # Parameters for the mean returns vector
    dict_params_mean_true = {VEC_MU: rng.uniform(0, 1, dim) / 50}

    # Params for z \sim SGT
    dict_params_z_true = {
        VEC_LBDA: rng.uniform(-0.25, 0.25, dim),
        VEC_P0: rng.uniform(2, 4, dim),
        VEC_Q0: rng.uniform(2, 4, dim),
    }

    # Params for DCC -- univariate vols
    dict_params_dcc_uvar_vol_true = {
        VEC_OMEGA: rng.uniform(0, 1, dim) / 2,
        VEC_BETA: rng.uniform(0, 1, dim) / 3,
        VEC_ALPHA: rng.uniform(0, 1, dim) / 10,
        VEC_PSI: rng.uniform(0, 1, dim) / 5,
    }
    # Params for DCC -- multivariate correlations
    dict_params_dcc_mvar_cor_true = {
        # Ensure \delta_1, \delta_2 \in [0,1] and \delta_1 + \delta_2 \le 1
        VEC_DELTA: np.array([0.007, 0.930]),
        MAT_QBAR: _generate_random_cov_mat(key=key, dim=dim) / 5,
    }

    dict_params_true = {
        DICT_PARAMS_MEAN: dict_params_mean_true,
        DICT_PARAMS_Z: dict_params_z_true,
        DICT_PARAMS_DCC_UVAR_VOL: dict_params_dcc_uvar_vol_true,
        DICT_PARAMS_DCC_MVAR_COR: dict_params_dcc_mvar_cor_true,
    }

    data_z, mat_returns = simulate_returns(
        seed=seed,
        dim=dim,
        num_sample=num_sample,
        dict_params_mean=dict_params_mean_true,
        dict_params_z=dict_params_z_true,
        dict_params_dcc_uvar_vol=dict_params_dcc_uvar_vol_true,
        dict_params_dcc_mvar_cor=dict_params_dcc_mvar_cor_true,
        num_cores=num_cores,
    )

    dict_params_mean_init_guess = {"vec_mu": rng.uniform(0, 1, dim) / 50}
    dict_params_z_init_guess = {
        VEC_LBDA: rng.uniform(-0.25, 0.25, dim),
        VEC_P0: rng.uniform(2, 4, dim),
        VEC_Q0: rng.uniform(2, 4, dim),
    }
    dict_params_dcc_uvar_vol_init_guess = {
        VEC_OMEGA: rng.uniform(0, 1, dim) / 2,
        VEC_BETA: rng.uniform(0, 1, dim) / 3,
        VEC_ALPHA: rng.uniform(0, 1, dim) / 10,
        VEC_PSI: rng.uniform(0, 1, dim) / 5,
    }
    # Params for DCC -- multivariate correlations
    dict_params_dcc_mvar_cor_init_guess = {
        VEC_DELTA: np.array([0.05, 0.530]),
        MAT_QBAR: _generate_random_cov_mat(key=key, dim=dim) / 10,
    }

    dict_params_init_guess = {
        DICT_PARAMS_MEAN: dict_params_mean_init_guess,
        DICT_PARAMS_Z: dict_params_z_init_guess,
        DICT_PARAMS_DCC_UVAR_VOL: dict_params_dcc_uvar_vol_init_guess,
        DICT_PARAMS_DCC_MVAR_COR: dict_params_dcc_mvar_cor_init_guess,
    }

    # Initial {\sigma_{i,0}}
    key, _ = random.split(key)
    mat_Sigma_0 = _generate_random_cov_mat(key=key, dim=dim)
    vec_sigma_0 = jnp.sqrt(jnp.diag(mat_Sigma_0))

    # Initial Q_0
    key, _ = random.split(key)
    mat_Q_0 = _generate_random_cov_mat(key=key, dim=dim)

    method = "BFGS"
    options = {"maxiter": 1000, "gtol": 1e-4}

    dict_init_t0_conditions = {VEC_SIGMA_0: vec_sigma_0, MAT_Q_0: mat_Q_0}

    neg_loglik_optval, dict_params = dcc_garch_optimization(
        mat_returns=mat_returns,
        dict_params_init_guess=dict_params_init_guess,
        dict_init_t0_conditions=dict_init_t0_conditions,
    )
