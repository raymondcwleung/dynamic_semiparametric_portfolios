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
import jaxtyping as jpt

import chex

import logging
import os
import pathlib
import pickle

from dataclasses import dataclass

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="logs/dcc.log",
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
from sgt import ParamsZSgt, SimulatedInnovations

# HACK:
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod
# jax.config.update("jax_debug_nans", True)  # Should disable in full prod


@chex.dataclass
class ParamsMean:
    """
    Parameters for the mean returns in a DCC-GARCH model
    """

    vec_mu: jpt.Float[jpt.Array, "dim"]


@chex.dataclass
class ParamsUVarVol:
    """
    Parameters for the univariate volatilities in a DCC-GARCH model
    """

    vec_omega: jpt.Float[jpt.Array, "dim"]
    vec_beta: jpt.Float[jpt.Array, "dim"]
    vec_alpha: jpt.Float[jpt.Array, "dim"]
    vec_psi: jpt.Float[jpt.Array, "dim"]

    def __post_init__(self):
        # All \omega, \beta, \alpha, \psi quantities must be
        # strictly positive
        if jnp.any(self.vec_omega < 0):
            raise ValueError("Must have positive-valued \\omega entries")

        if jnp.any(self.vec_beta < 0):
            raise ValueError("Must have positive-valued \\beta entries")

        if jnp.any(self.vec_alpha < 0):
            raise ValueError("Must have positive-valued \\alpha entries")

        if jnp.any(self.vec_psi < 0):
            raise ValueError("Must have positive-valued \\psi entries")


@chex.dataclass
class ParamsMVarCor:
    """
    Parameters for the multivariate Q_t process in a DCC-GARCH model
    """

    vec_delta: jpt.Float[jpt.Array, "2"]
    mat_Qbar: jpt.Float[jpt.Array, "dim dim"]

    def __post_init__(self):
        # Check constraints on \delta
        if jnp.size(self.vec_delta) != 2:
            raise ValueError("\\delta must have exactly two elements")

        if jnp.any(self.vec_delta < 0) or jnp.any(self.vec_delta > 1):
            raise ValueError("All \\delta parameter entries must be in (0,1)")

        if jnp.sum(self.vec_delta) > 1:
            raise ValueError(
                "Sum of \\delta[0] + \\delta[1] must be strictly less than 1"
            )

        # Check constraints on \bar{Q}
        eigvals, _ = jnp.linalg.eigh(self.mat_Qbar)
        if jnp.any(eigvals < 0):
            raise ValueError("\\bar{Q} must be PSD")


@chex.dataclass
class ParamsDcc:
    """
    Collect all the parameters of a DCC-GARCH model.
    """

    mean: ParamsMean
    uvar_vol: ParamsUVarVol
    mvar_cor: ParamsMVarCor


@chex.dataclass
class ParamsModel:
    """
    Generic parent class for holding the parameters of an entire DCC-X-GARCH
    model, where X depends on the choice of specifications of the
    innovation process z_t.
    """

    dcc: ParamsDcc


@chex.dataclass
class ParamsDccSgtGarch(ParamsModel):
    """
    Collect all the parameters of a DCC-SGT-GARCH model
    """

    sgt: sgt.ParamsZSgt


@chex.dataclass
class InitTimeConditionDcc:
    """
    Initial conditions related to the process Q_t.
    NOTE: This is primarily only used for simulating data and
    not on real data.
    """

    mat_Sigma_init_t0: jpt.Float[jpt.Array, "dim dim"]
    mat_Q_init_t0: jpt.Float[jpt.Array, "dim dim"]

    def __post_init__(self):
        self.vec_sigma_init_t0 = jnp.sqrt(jnp.diag(self.mat_Sigma_init_t0))


@chex.dataclass
class InitTimeConditionDccSgtGarch:
    sgt: sgt.InitTimeConditionZSgt
    dcc: InitTimeConditionDcc


logger = logging.getLogger(__name__)

NUM_LBDA_TVPARAMS = 3
NUM_P0_TVPARAMS = 3
NUM_Q0_TVPARAMS = 3

# DICT_INIT_T0_CONDITIONS = "dict_init_t0_conditions"
# DICT_PARAMS_MEAN = "dict_params_mean"
# DICT_PARAMS_Z = "dict_params_z"
# DICT_PARAMS_DCC_UVAR_VOL = "dict_params_dcc_uvar_vol"
# DICT_PARAMS_DCC_MVAR_COR = "dict_params_dcc_mvar_cor"

# # Time t = 0 initial conditions
# VEC_SIGMA_0 = "vec_sigma_0"
# MAT_Q_0 = "mat_Q_0"
#
# # SGT parameters
# VEC_LBDA = "vec_lbda"
# VEC_P0 = "vec_p0"
# VEC_Q0 = "vec_q0"
#
# # Mean return parameters
# VEC_MU = "vec_mu"
#
# # Univariate volatilities parameters
# VEC_OMEGA = "vec_omega"
# VEC_BETA = "vec_beta"
# VEC_ALPHA = "vec_alpha"
# VEC_PSI = "vec_psi"
#
# # Multivariate DCC parameters
# VEC_DELTA = "vec_delta"
# MAT_QBAR = "mat_Qbar"


@dataclass
class SimulatedReturns:
    """
    Data class for keeping track of the parameters and data
    of simulated returns.
    """

    # Dimension (i.e. number of assets) and number of time samples
    dim: int
    num_sample: int

    ################################################################
    ## Object for the innovations z_t
    ################################################################
    siminnov: sgt.SimulatedInnovations

    ################################################################
    ## Parameters
    ################################################################
    params_dcc_true: ParamsDcc

    ################################################################
    ## Simulated data
    ################################################################
    # Unexpected excess returns \epsilon_t
    data_mat_epsilon: jpt.ArrayLike

    # Univariate volatilities \sigma_{i,t}'s
    data_mat_sigma: jpt.ArrayLike

    # Normalized unexpected returns u_t
    data_mat_u: jpt.ArrayLike

    # DCC parameters Q_t
    data_tns_Q: jpt.ArrayLike

    # DCC covariances \Sigma_t
    data_tns_Sigma: jpt.ArrayLike

    # Simulated asset returns
    data_mat_returns: jpt.ArrayLike


def generate_random_cov_mat(
    key: KeyArrayLike, dim: int
) -> jpt.Float[jpt.Array, "dim dim"]:
    """
    Generate a random covariance-variance matrix (i.e. symmetric and PSD).
    """
    mat_A = jax.random.normal(key, shape=(dim, dim)) * 0.25
    mat_sigma = jnp.transpose(mat_A) @ mat_A

    return mat_sigma


def _calc_mean_return(
    num_sample: int, dim: int, vec_mu: jpt.Float[jpt.Array, "dim"]
) -> jpt.Float[jpt.Array, "num_sample dim"]:
    """
    Compute the mean returns \\mu
    """
    mat_mu = jnp.tile(vec_mu, num_sample).reshape(num_sample, dim)
    return mat_mu


def _calc_demean_returns(
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"],
    vec_mu: jpt.Float[jpt.Array, "dim"],
) -> jpt.Float[jpt.Array, "num_sample dim"]:
    """
    Calculate \\epsilon_t = R_t - \\mu_t
    """
    mat_epsilon = mat_returns - vec_mu
    return mat_epsilon


@jax.jit
def _calc_unexpected_excess_rtn(
    mat_Sigma: jpt.Float[jpt.Array, "num_sample dim"],
    vec_z: jpt.Float[jpt.Array, "dim"],
) -> jpt.Float[jpt.Array, "num_sample dim"]:
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
    vec_sigma: jpt.Float[jpt.Array, "dim"], vec_epsilon: jpt.Float[jpt.Array, "dim"]
) -> jpt.Float[jpt.Array, "dim"]:
    """
    Return u_t = D_t^{-1} \\epsilon_t
    """
    mat_inv_D = jnp.diag(1 / vec_sigma)
    vec_u = mat_inv_D @ vec_epsilon

    return vec_u


@jax.jit
def _calc_mat_Q(
    vec_delta: jpt.Float[jpt.Array, "dim"],
    vec_u_t_minus_1: jpt.Float[jpt.Array, "dim"],
    mat_Q_t_minus_1: jpt.Float[jpt.Array, "dim dim"],
    mat_Qbar: jpt.Float[jpt.Array, "dim dim"],
) -> jpt.Float[jpt.Array, "dim dim"]:
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
def _calc_mat_Gamma(
    mat_Q: jpt.Float[jpt.Array, "dim dim"]
) -> jpt.Float[jpt.Array, "dim dim"]:
    """
    Return the \\Gamma_t matrix of the DCC model
    """
    mat_Qstar_inv_sqrt = jnp.diag(jnp.diag(mat_Q) ** (-1 / 2))
    mat_Gamma = mat_Qstar_inv_sqrt @ mat_Q @ mat_Qstar_inv_sqrt

    return mat_Gamma


@jax.jit
def _calc_mat_Sigma(
    vec_sigma: jpt.Float[jpt.Array, "dim"], mat_Gamma: jpt.Float[jpt.Array, "dim dim"]
) -> jpt.Float[jpt.Array, "dim dim"]:
    """
    Return the covariance \\Sigma_t = D_t \\Gamma_t D_t,
    where D_t is a diagonal matrix of \\sigma_{i,t}.
    """
    mat_D = jnp.diag(vec_sigma)

    mat_Sigma = mat_D @ mat_Gamma @ mat_D
    return mat_Sigma


def simulate_dcc(
    data_z: jpt.Float[jpt.Array, "num_sample dim"],
    params_dcc_true: ParamsDcc,
    inittimecond_dcc: InitTimeConditionDcc,
) -> tp.Tuple[
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_epsilon
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_sigma
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_u
    jpt.Float[jpt.Array, "num_sample dim dim"],  # tns_Q
    jpt.Float[jpt.Array, "num_sample dim dim"],  # tns_Sigma
]:
    """
    Simulate a DCC model
    """
    logger.info("Begin DCC simulation.")

    vec_omega = params_dcc_true.uvar_vol.vec_omega
    vec_beta = params_dcc_true.uvar_vol.vec_beta
    vec_alpha = params_dcc_true.uvar_vol.vec_alpha
    vec_psi = params_dcc_true.uvar_vol.vec_psi

    vec_delta = params_dcc_true.mvar_cor.vec_delta
    mat_Qbar = params_dcc_true.mvar_cor.mat_Qbar

    mat_Sigma_init_t0 = inittimecond_dcc.mat_Sigma_init_t0
    mat_Q_init_t0 = inittimecond_dcc.mat_Q_init_t0

    num_sample = jnp.shape(data_z)[0]
    dim = jnp.shape(data_z)[1]

    # Initial conditions at t = 0
    vec_z_0 = data_z[0, :]
    vec_sigma_init_t0 = jnp.sqrt(jnp.diag(mat_Sigma_init_t0))
    vec_epsilon_init_t0 = _calc_unexpected_excess_rtn(
        mat_Sigma=mat_Sigma_init_t0, vec_z=vec_z_0
    )
    vec_u_init_t0 = _calc_normalized_unexpected_excess_rtn(
        vec_sigma=vec_sigma_init_t0, vec_epsilon=vec_epsilon_init_t0
    )

    # Init
    lst_epsilon = [jnp.empty(dim)] * num_sample
    lst_sigma = [jnp.empty(dim)] * num_sample
    lst_u = [jnp.empty(dim)] * num_sample
    lst_Q = [jnp.empty((dim, dim))] * num_sample
    lst_Sigma = [jnp.empty((dim, dim))] * num_sample

    # Save initial conditions
    lst_epsilon[0] = vec_epsilon_init_t0
    lst_sigma[0] = vec_sigma_init_t0
    lst_u[0] = vec_u_init_t0
    lst_Q[0] = mat_Q_init_t0
    lst_Sigma[0] = mat_Sigma_init_t0

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
    mat_sigma = jnp.array(lst_sigma)
    mat_u = jnp.array(lst_u)
    tns_Q = jnp.array(lst_Q)
    tns_Sigma = jnp.array(lst_Sigma)
    return mat_epsilon, mat_sigma, mat_u, tns_Q, tns_Sigma


@jax.jit
def _calc_asymmetric_garch_sigma2(
    vec_sigma_t_minus_1: jpt.Float[jpt.Array, "dim"],
    vec_epsilon_t_minus_1: jpt.Float[jpt.Array, "dim"],
    vec_omega: jpt.Float[jpt.Array, "dim"],
    vec_beta: jpt.Float[jpt.Array, "dim"],
    vec_alpha: jpt.Float[jpt.Array, "dim"],
    vec_psi: jpt.Float[jpt.Array, "dim"],
) -> jpt.Float[jpt.Array, "dim"]:
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
    mat_epsilon: jpt.Float[jpt.Array, "num_sample dim"],
    vec_sigma_init_t0: jpt.Float[jpt.Array, "dim"],
    vec_omega: jpt.Float[jpt.Array, "dim"],
    vec_beta: jpt.Float[jpt.Array, "dim"],
    vec_alpha: jpt.Float[jpt.Array, "dim"],
    vec_psi: jpt.Float[jpt.Array, "dim"],
) -> jpt.Float[jpt.Array, "num_sample dim"]:
    """
    Calculate the trajectory of univariate vol's {\\sigma_{i,t}}
    for all t based on the asymmetric-GARCH(1,1) model
    """
    num_sample = mat_epsilon.shape[0]
    dim = mat_epsilon.shape[1]

    mat_sigma = jnp.empty(shape=(num_sample, dim))
    mat_sigma = mat_sigma.at[0].set(vec_sigma_init_t0)

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
    mat_epsilon: jpt.Float[jpt.Array, "num_sample dim"],
    mat_sigma: jpt.Float[jpt.Array, "num_sample dim"],
    mat_Q_0: jpt.Float[jpt.Array, "num_sample dim"],
    vec_delta: jpt.Float[jpt.Array, "dim"],
    mat_Qbar: jpt.Float[jpt.Array, "num_sample dim"],
) -> jpt.Float[jpt.Array, "num_sample dim dim"]:
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
    mat_sigma: jpt.Float[jpt.Array, "num_sample dim"],
    mat_epsilon: jpt.Float[jpt.Array, "num_sample dim"],
) -> jpt.Float[jpt.Array, "num_sample dim"]:
    """
    Calculate the trajectory of u_t = D_t^{-1}\\epsilon_t
    """
    mat_u = mat_epsilon / mat_sigma
    return mat_u


def _calc_innovations(
    vec_epsilon: jpt.Float[jpt.Array, "dim"], mat_Sigma: jpt.Float[jpt.Array, "dim dim"]
) -> jpt.Float[jpt.Array, "dim"]:
    """
    Return innovations z_t = \\Sigma_t^{-1/2} \\epsilon_t, where we are
    given \\epsilon_t = R_t - \\mu_t and conditional covariances
    {\\Sigma_t}
    """
    mat_Sigma_sqrt = jnp.linalg.cholesky(mat_Sigma, upper=True)
    vec_z = jnp.linalg.solve(mat_Sigma_sqrt, vec_epsilon)

    return vec_z


def _calc_trajectory_innovations(
    mat_epsilon: jpt.Float[jpt.Array, "num_sample dim"],
    tns_Sigma: jpt.Float[jpt.Array, "num_sample dim dim"],
) -> jpt.Float[jpt.Array, "num_sample dim"]:
    """
    Calculate trajectory of innovations z_t's over the full sample.
    """
    _func = vmap(_calc_innovations, in_axes=[0, 0])
    mat_z = _func(mat_epsilon, tns_Sigma)

    return mat_z


def _calc_trajectory_innovations_timevarying_lbda(
    mat_lbda_tvparams: jpt.Float[jpt.Array, "NUM_LBDA_TVPARAMS dim"],
    mat_z: jpt.Float[jpt.Array, "num_sample dim"],
    vec_lbda_init_t0: jpt.Float[jpt.Array, "dim"],
) -> jpt.Float[jpt.Array, "num_sample dim"]:
    """
    Calculate the time-varying \\lambda_t parameters associated with the
    innovations z_t
    """
    num_sample = mat_z.shape[0]
    dim = mat_z.shape[1]

    mat_lbda = jnp.empty(shape=(num_sample, dim))
    mat_lbda = mat_lbda.at[0].set(vec_lbda_init_t0)

    _func_lbda = vmap(sgt._time_varying_lbda_params, in_axes=[1, 0, 0])

    def _body_fun(tt, mat_lbda):
        vec_lbda_t = _func_lbda(mat_lbda_tvparams, mat_lbda[tt - 1], mat_z[tt - 1, :])
        mat_lbda = mat_lbda.at[tt].set(vec_lbda_t)
        return mat_lbda

    mat_lbda = jax.lax.fori_loop(
        lower=1, upper=num_sample, body_fun=_body_fun, init_val=mat_lbda
    )
    return mat_lbda


def _calc_trajectory_innovations_timevarying_pq(
    mat_pq_tvparams: jpt.Float[jpt.Array, "?num_pq_tvparams dim"],
    mat_z: jpt.Float[jpt.Array, "num_sample dim"],
    vec_pq_init_t0: jpt.Float[jpt.Array, "dim"],
) -> jpt.Float[jpt.Array, "num_sample dim"]:
    """
    Calculate the time-varying p_t or q_t parameters associated with the
    innovations z_t
    """
    num_sample = mat_z.shape[0]
    dim = mat_z.shape[1]

    mat = jnp.empty(shape=(num_sample, dim))
    mat = mat.at[0].set(vec_pq_init_t0)

    _func_lbda = vmap(sgt._time_varying_pq_params, in_axes=[1, 0, 0])

    def _body_fun(tt, mat_lbda):
        vec_lbda_t = _func_lbda(mat_pq_tvparams, mat_lbda[tt - 1], mat_z[tt - 1, :])
        mat_lbda = mat_lbda.at[tt].set(vec_lbda_t)
        return mat_lbda

    mat = jax.lax.fori_loop(lower=1, upper=num_sample, body_fun=_body_fun, init_val=mat)
    return mat


def calc_trajectories(
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"],
    params_dcc_sgt_garch: ParamsDccSgtGarch,
    inittimecond_dcc_sgt_garch: InitTimeConditionDccSgtGarch,
) -> tp.Tuple[
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_lbda
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_p0
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_q0
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_epsilon
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_sigma
    jpt.Float[jpt.Array, "num_sample dim dim"],  # tns_Sigma
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_z
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_u
]:
    """
    Given parameters, return the trajectories {\\lambda_t}, {p_t}, {q_t},
    {\\epsilon_t}, {\\sigma_{i,t}}, {\\Sigma_t}, {z_t}, {u_t}
    """
    # Extract the parameters and t = 0 initial conditions
    mat_lbda_tvparams = params_dcc_sgt_garch.sgt.mat_lbda_tvparams
    mat_p0_tvparams = params_dcc_sgt_garch.sgt.mat_p0_tvparams
    mat_q0_tvparams = params_dcc_sgt_garch.sgt.mat_p0_tvparams

    vec_mu = params_dcc_sgt_garch.dcc.mean.vec_mu

    vec_omega = params_dcc_sgt_garch.dcc.uvar_vol.vec_omega
    vec_beta = params_dcc_sgt_garch.dcc.uvar_vol.vec_beta
    vec_alpha = params_dcc_sgt_garch.dcc.uvar_vol.vec_alpha
    vec_psi = params_dcc_sgt_garch.dcc.uvar_vol.vec_psi

    vec_delta = params_dcc_sgt_garch.dcc.mvar_cor.vec_delta
    mat_Qbar = params_dcc_sgt_garch.dcc.mvar_cor.mat_Qbar

    vec_lbda_init_t0 = inittimecond_dcc_sgt_garch.sgt.vec_lbda_init_t0
    vec_p0_init_t0 = inittimecond_dcc_sgt_garch.sgt.vec_p0_init_t0
    vec_q0_init_t0 = inittimecond_dcc_sgt_garch.sgt.vec_q0_init_t0

    vec_sigma_init_t0 = inittimecond_dcc_sgt_garch.dcc.vec_sigma_init_t0
    mat_Q_init_t0 = inittimecond_dcc_sgt_garch.dcc.mat_Q_init_t0

    # Compute \epsilon_t = R_t - \mu_t
    mat_epsilon = _calc_demean_returns(mat_returns=mat_returns, vec_mu=vec_mu)

    # Calculate the univariate vol's \sigma_{i,t}'s
    mat_sigma = _calc_trajectory_uvar_vol(
        mat_epsilon=mat_epsilon,
        vec_sigma_init_t0=vec_sigma_init_t0,
        vec_omega=vec_omega,
        vec_beta=vec_beta,
        vec_alpha=vec_alpha,
        vec_psi=vec_psi,
    )

    # Calculate the multivariate covariance \Sigma_t
    tns_Sigma = _calc_trajectory_mvar_cov(
        mat_epsilon=mat_epsilon,
        mat_sigma=mat_sigma,
        mat_Q_0=mat_Q_init_t0,
        vec_delta=vec_delta,
        mat_Qbar=mat_Qbar,
    )

    # Calculate the innovations z_t = \Sigma_t^{-1/2} \epsilon_t
    mat_z = _calc_trajectory_innovations(mat_epsilon=mat_epsilon, tns_Sigma=tns_Sigma)

    # Calculate the normalized unexpected returns u_t
    mat_u = _calc_trajectory_normalized_unexp_returns(
        mat_sigma=mat_sigma, mat_epsilon=mat_epsilon
    )

    # Calculate the time-varying parameters \lambda_t
    # associated with the innovations z_t
    mat_lbda = _calc_trajectory_innovations_timevarying_lbda(
        mat_lbda_tvparams=mat_lbda_tvparams,
        mat_z=mat_z,
        vec_lbda_init_t0=vec_lbda_init_t0,
    )

    # Calculate the time-varying parameters p_t
    # associated with the innovations z_t
    mat_p0 = _calc_trajectory_innovations_timevarying_pq(
        mat_pq_tvparams=mat_p0_tvparams, mat_z=mat_z, vec_pq_init_t0=vec_p0_init_t0
    )

    # Calculate the time-varying parameters q_t
    # associated with the innovations z_t
    mat_q0 = _calc_trajectory_innovations_timevarying_pq(
        mat_pq_tvparams=mat_q0_tvparams, mat_z=mat_z, vec_pq_init_t0=vec_q0_init_t0
    )

    return mat_lbda, mat_p0, mat_q0, mat_epsilon, mat_sigma, tns_Sigma, mat_z, mat_u


def simulate_dcc_sgt_garch(
    key: KeyArrayLike,
    dim: int,
    num_sample: int,
    # SGT
    params_z_sgt_true: sgt.ParamsZSgt,
    inittimecond_z_sgt: sgt.InitTimeConditionZSgt,
    # DCC
    params_dcc_true: ParamsDcc,
    inittimecond_dcc: InitTimeConditionDcc,
    # Saving paths
    data_simreturns_savepath: os.PathLike,
) -> SimulatedReturns:
    """
    Simulate DCC-SGT-GARCH.
    """
    # Get the simulated innovations z_t
    siminnov = sgt.sample_mvar_timevarying_sgt(
        key=key,
        num_sample=num_sample,
        params_z_sgt_true=params_z_sgt_true,
        inittimecond_z_sgt=inittimecond_z_sgt,
        save_path=None,
    )

    # Obtain the simulated asset returns R_t
    simreturns = _simulate_returns(
        dim=dim,
        num_sample=num_sample,
        siminnov=siminnov,
        params_dcc_true=params_dcc_true,
        inittimecond_dcc=inittimecond_dcc,
        data_simreturns_savepath=data_simreturns_savepath,
    )

    return simreturns


def _simulate_returns(
    dim: int,
    num_sample: int,
    siminnov: sgt.SimulatedInnovations,
    params_dcc_true: ParamsDcc,
    inittimecond_dcc: InitTimeConditionDcc,
    data_simreturns_savepath: os.PathLike,
) -> SimulatedReturns:
    """
    Simulate asset returns.

    In particular, this function allows for varying the innovations z_t specifications
    while keeping the DCC-GARCH structure. In other words, this function constructs a
    DCC-X-GARCH, where X depends on the specifications choice of z_t.
    """
    # Load in the innovations
    data_z = siminnov.data_mat_z

    # Sanity checks
    try:
        if data_z.shape[0] != num_sample:
            raise ValueError("Incorrect 'num_sample' for the simulated innovations.")

        if data_z.shape[1] != dim:
            raise ValueError("Incorrect 'dim' for the simulated innovations.")
    except Exception as e:
        logger.error(str(e))
        raise

    # Simulate a DCC model
    data_mat_epsilon, data_mat_sigma, data_mat_u, data_tns_Q, data_tns_Sigma = (
        simulate_dcc(
            data_z=data_z,
            params_dcc_true=params_dcc_true,
            inittimecond_dcc=inittimecond_dcc,
        )
    )

    # Set the asset mean
    mat_mu = _calc_mean_return(
        num_sample=num_sample, dim=dim, vec_mu=params_dcc_true.mean.vec_mu
    )

    # Asset returns
    data_mat_returns = mat_mu + data_mat_epsilon

    logger.info("Done simulating returns")

    # Save
    simreturns = SimulatedReturns(
        dim=dim,
        num_sample=num_sample,
        siminnov=siminnov,
        params_dcc_true=params_dcc_true,
        data_mat_epsilon=data_mat_epsilon,
        data_mat_sigma=data_mat_sigma,
        data_mat_u=data_mat_u,
        data_tns_Q=data_tns_Q,
        data_tns_Sigma=data_tns_Sigma,
        data_mat_returns=data_mat_returns,
    )
    with open(data_simreturns_savepath, "wb") as f:
        pickle.dump(simreturns, f)
        logger.info(f"Saved DCC simulations to {str(data_simreturns_savepath)}")

    return simreturns


def dcc_sgt_loglik(
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"],
    params_dcc_sgt_garch: ParamsDccSgtGarch,
    inittimecond_dcc_sgt_garch: InitTimeConditionDccSgtGarch,
    neg_loglik: bool = True,
) -> jpt.Float:
    """
    (Negative) of the likelihood of the
    DCC-time-varying-SGT-Asymmetric-GARCH(1,1) model
    """
    mat_lbda, mat_p0, mat_q0, _, _, tns_Sigma, mat_z, _ = calc_trajectories(
        mat_returns=mat_returns,
        params_dcc_sgt_garch=params_dcc_sgt_garch,
        inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch,
    )

    # Compute {\log\det \Sigma_t}
    _, vec_logdet_Sigma = jnp.linalg.slogdet(tns_Sigma)

    # Compute the log-likelihood of g(z_t | \theta_t)
    # where g \sim SGT
    sgt_loglik = sgt.loglik_mvar_timevarying_sgt(
        data=mat_z,
        mat_lbda=mat_lbda,
        mat_p0=mat_p0,
        mat_q0=mat_q0,
        neg_loglik=False,
    )

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
        raise

    dict_params_z = {
        VEC_LBDA: x[0:dim],
        VEC_P0: x[dim : 2 * dim],
        VEC_Q0: x[2 * dim :],
    }

    return dict_params_z


def _make_params_from_arr_z_sgt(x, dim) -> sgt.ParamsZSgt:
    """
    Take a vector x and split them into parameters related to the
    time-varying SGT process
    """
    try:
        if (
            jnp.size(x)
            != sgt.NUM_LBDA_TVPARAMS * dim
            + sgt.NUM_P0_TVPARAMS * dim
            + sgt.NUM_Q0_TVPARAMS * dim
        ):
            raise ValueError(
                "Incorrect total number of parameters for time-varying SGT innovations specification"
            )
    except Exception as e:
        logger.error(str(e))
        raise

    mat_lbda_tvparams = x[0 : (sgt.NUM_LBDA_TVPARAMS * dim)].reshape(
        sgt.NUM_LBDA_TVPARAMS, dim
    )
    mat_p0_tvparams = x[
        (sgt.NUM_LBDA_TVPARAMS * dim) : (
            sgt.NUM_LBDA_TVPARAMS * dim + sgt.NUM_P0_TVPARAMS * dim
        )
    ].reshape(sgt.NUM_P0_TVPARAMS, dim)
    mat_q0_tvparams = x[
        (sgt.NUM_LBDA_TVPARAMS * dim + sgt.NUM_P0_TVPARAMS * dim) :
    ].reshape(sgt.NUM_Q0_TVPARAMS, dim)

    params_z_sgt = sgt.ParamsZSgt(
        mat_lbda_tvparams=mat_lbda_tvparams,
        mat_p0_tvparams=mat_p0_tvparams,
        mat_q0_tvparams=mat_q0_tvparams,
    )
    return params_z_sgt


def _make_params_from_arr_mean(x, dim) -> ParamsMean:
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
        raise

    params_mean = ParamsMean(vec_mu=x)
    return params_mean


def _make_params_from_arr_dcc_uvar_vol(x, dim) -> ParamsUVarVol:
    """
    Take a vector x and split them into parameters related to the
    univariate GARCH processes.
    """
    params_uvar_vol = ParamsUVarVol(
        vec_omega=x[0:dim],
        vec_beta=x[dim : 2 * dim],
        vec_alpha=x[2 * dim : 3 * dim],
        vec_psi=x[3 * dim :],
    )
    return params_uvar_vol


# def _make_dict_params_mean(x, dim) -> tp.Dict[str, NDArray]:
#     """
#     Take a vector x and split them into parameters related to the
#     mean \\mu
#     """
#     try:
#         if jnp.size(x) != dim:
#             raise ValueError(
#                 "Total number of parameters for the constant mean process is incorrect."
#             )
#     except Exception as e:
#         logger.error(str(e))
#         raise
#
#     dict_params_mean = {VEC_MU: x}
#
#     return dict_params_mean


# def _make_dict_params_dcc_uvar_vol(x, dim) -> tp.Dict[str, NDArray]:
#     """
#     Take a vector x and split them into parameters related to the
#     univariate GARCH processes.
#     """
#     dict_params_dcc_uvar_vol = {
#         VEC_OMEGA: x[0:dim],
#         VEC_BETA: x[dim : 2 * dim],
#         VEC_ALPHA: x[2 * dim : 3 * dim],
#         VEC_PSI: x[3 * dim :],
#     }
#
#     return dict_params_dcc_uvar_vol


def _make_params_from_arr_dcc_mvar_cor(
    x,
    dim,
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"],
    params_dcc_sgt_garch: ParamsDccSgtGarch,
    inittimecond_dcc_sgt_garch: InitTimeConditionDccSgtGarch,
) -> ParamsMVarCor:
    """
    Take a vector x and split them into parameters related to the
    DCC Q_t process.
    """
    try:
        if jnp.size(x) != 2:
            raise ValueError(
                "Total number of parameters for the DCC process is incorrect."
            )
    except Exception as e:
        logger.error(str(e))
        raise

    vec_delta = x

    # Special treatment estimating \bar{Q}. Apply the ``variance-
    # targetting'' approach. That is, we estimate by
    # \hat{\bar{Q}} = sample moment of \hat{u}_t\hat{u}_t^{\top}.
    _, _, _, _, _, _, _, mat_u = calc_trajectories(
        mat_returns=mat_returns,
        params_dcc_sgt_garch=params_dcc_sgt_garch,
        inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch,
    )

    # Set \hat{\bar{Q}} = \frac{1}{T} \sum_t u_tu_t^{\top}
    _func = vmap(jnp.outer, in_axes=[0, 0])
    tens_uuT = _func(mat_u, mat_u)
    mat_Qbar = tens_uuT.mean(axis=0)

    try:
        if mat_Qbar.shape != (dim, dim):
            raise ValueError("Shape of estimated \\bar{Q} is incorrect.")
    except Exception as e:
        logger.error(str(e))
        raise

    params_mvar_cor = ParamsMVarCor(vec_delta=vec_delta, mat_Qbar=mat_Qbar)
    return params_mvar_cor


# def _make_dict_params_dcc_mvar_cor(
#     x,
#     dim,
#     mat_returns,
#     dict_init_t0_conditions,
#     dict_params_mean,
#     dict_params_z,  # Unused
#     dict_params_dcc_uvar_vol,
#     dict_params_dcc_mvar_cor,
# ) -> tp.Dict[str, NDArray]:
#     """
#     Take a vector x and split them into parameters related to the
#     DCC process.
#     """
#     try:
#         if jnp.size(x) != 2:
#             raise ValueError(
#                 "Total number of parameters for the DCC process is incorrect."
#             )
#     except Exception as e:
#         logger.error(str(e))
#         raise
#
#     dict_params_dcc_mvar_cor[VEC_DELTA] = x
#
#     # Special treatment estimating \bar{Q}. Apply the ``variance-
#     # targetting'' approach. That is, we estimate by
#     # \hat{\bar{Q}} = sample moment of \hat{u}_t\hat{u}_t^{\top}.
#     _, _, _, _, mat_u = calc_trajectories(
#         mat_returns=mat_returns,
#         dict_init_t0_conditions=dict_init_t0_conditions,
#         dict_params_mean=dict_params_mean,
#         dict_params_dcc_uvar_vol=dict_params_dcc_uvar_vol,
#         dict_params_dcc_mvar_cor=dict_params_dcc_mvar_cor,
#     )
#
#     # Set \hat{\bar{Q}} = \frac{1}{T} \sum_t u_tu_t^{\top}
#     _func = vmap(jnp.outer, in_axes=[0, 0])
#     tens_uuT = _func(mat_u, mat_u)
#     mat_Qbar = tens_uuT.mean(axis=0)
#     dict_params_dcc_mvar_cor[MAT_QBAR] = mat_Qbar
#
#     return dict_params_dcc_mvar_cor


def _objfun_dcc_loglik(
    x,
    make_dict_params_fun,
    params: ParamsDccSgtGarch,
    # dict_params: tp.Dict[str, jpt.Array],
    # dict_init_t0_conditions: tp.Dict[str, jpt.Array],
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"],
) -> jpt.DTypeLike:
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
        neg_loglik = dcc_sgt_loglik(
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
    dict_params: tp.Dict[str, jpt.Array]
) -> jpt.Array:
    """
    Take in a dictionary of parameters and flatten them to an
    array in preparation for optimization.
    """
    x0 = itertools.chain.from_iterable(dict_params.values())
    x0 = list(x0)
    x0 = np.array(x0)
    x0 = jnp.array(x0)

    return x0


def dcc_sgt_garch_optimization(
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"],
    params_init_guess: ParamsModel,
    # dict_params_init_guess,
    # dict_init_t0_conditions,
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
    Run DCC-SGT-GARCH optimization
    """
    logger.info(f"Begin DCC-SGT-GARCH optimization.")

    # solver_obj_z = solver_z(
    #     fun=objfun_dcc_loglik_opt_params_z,
    #     verbose=verbose,
    #     maxiter=maxiter,
    #     tol=tol,
    # )
    #     sol_z = solver_obj_z.run(init_params=x0_z, dict_params=dict_params)

    #################################################################
    ## Setup partial objective functions
    #################################################################
    objfun_dcc_loglik_opt_params_z = lambda x, params: _objfun_dcc_loglik(
        x=x,
        params=params,
        make_dict_params_fun=_make_dict_params_timevarying_sgt,
        # dict_init_t0_conditions=dict_init_t0_conditions,
        mat_returns=mat_returns,
    )

    # objfun_dcc_loglik_opt_params_mean = lambda x, dict_params: _objfun_dcc_loglik(
    #     x=x,
    #     dict_params=dict_params,
    #     make_dict_params_fun=_make_dict_params_mean,
    #     dict_init_t0_conditions=dict_init_t0_conditions,
    #     mat_returns=mat_returns,
    # )
    #
    # objfun_dcc_loglik_opt_params_dcc_uvar_vol = (
    #     lambda x, dict_params: _objfun_dcc_loglik(
    #         x=x,
    #         dict_params=dict_params,
    #         make_dict_params_fun=_make_dict_params_dcc_uvar_vol,
    #         dict_init_t0_conditions=dict_init_t0_conditions,
    #         mat_returns=mat_returns,
    #     )
    # )
    #
    # objfun_dcc_loglik_opt_params_dcc_mvar_cor = (
    #     lambda x, dict_params: _objfun_dcc_loglik(
    #         x=x,
    #         dict_params=dict_params,
    #         make_dict_params_fun=_make_dict_params_dcc_mvar_cor,
    #         dict_init_t0_conditions=dict_init_t0_conditions,
    #         mat_returns=mat_returns,
    #     )
    # )

    neg_loglik_optval = None
    params = params_init_guess
    for iter in range(grand_maxiter):
        #################################################################
        ## Step 1: Optimize for the parameters of z_t
        #################################################################
        x0_z = _make_params_from_arr_z_sgt(params)
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
    pass
