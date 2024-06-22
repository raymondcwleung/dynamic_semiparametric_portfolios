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

import dataclasses
from dataclasses import dataclass

from datetime import datetime


current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"logs/{current_time}_dcc.log",
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
from sgt import ParamsZSgt, SimulatedInnovations, loglik_mvar_timevarying_sgt

# HACK:
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod
jax.config.update("jax_debug_nans", True)  # Should disable in full prod

@chex.dataclass
class ParamsMean:
    """
    Parameters for the mean returns in a DCC-GARCH model
    """

    vec_mu: jpt.Float[jpt.Array, "dim"]

    def check_constraints(self) -> bool:
        # No particular constraints on the mean
        valid = True
        return valid


@chex.dataclass
class ParamsUVarVol:
    """
    Parameters for the univariate volatilities in a DCC-GARCH model
    """

    vec_alpha: jpt.Float[jpt.Array, "dim"]
    vec_beta: jpt.Float[jpt.Array, "dim"]
    vec_omega: jpt.Float[jpt.Array, "dim"]
    vec_psi: jpt.Float[jpt.Array, "dim"]

    def check_constraints(self) -> jpt.Bool:
        # All \omega, \beta, \alpha, \psi quantities must be
        # strictly positive

        valid_constraints_alpha = jax.lax.select(
            pred=jnp.all(self.vec_alpha > 0),
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )

        valid_constraints_beta = jax.lax.select(
            pred=jnp.all(self.vec_beta > 0),
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )

        valid_constraints_omega = jax.lax.select(
            pred=jnp.all(self.vec_omega > 0),
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )

        valid_constraints_psi = jax.lax.select(
            pred=jnp.all(self.vec_psi > 0),
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )

        vec_constraints = jnp.array(
            [
                valid_constraints_alpha,
                valid_constraints_beta,
                valid_constraints_omega,
                valid_constraints_psi,
            ]
        )
        valid_constraints = jax.lax.select(
            pred=jnp.all(vec_constraints),
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )
        # return valid_constraints
        # HACK::
        return True


@chex.dataclass
class ParamsMVarCor:
    """
    Parameters for the multivariate Q_t process in a DCC-GARCH model
    """

    vec_delta: jpt.Float[jpt.Array, "2"]
    mat_Qbar: jpt.Float[jpt.Array, "dim dim"]

    def check_constraints(self) -> jpt.Bool:

        # Check constraints on \delta
        valid_constraints_vec_delta_size = jax.lax.select(
            pred=jnp.size(self.vec_delta) == 2,
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )

        valid_constraints_vec_delta_range_lo = jax.lax.select(
            pred=jnp.all(self.vec_delta > 0),
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )

        valid_constraints_vec_delta_range_hi = jax.lax.select(
            pred=jnp.all(self.vec_delta < 1),
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )

        valid_constraints_vec_delta_sum = jax.lax.select(
            pred=jnp.sum(self.vec_delta) < 1,
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )

        # Check constraints on \bar{Q}
        eigvals, _ = jnp.linalg.eigh(self.mat_Qbar)
        valid_constraints_mat_Qbar_psd = jax.lax.select(
            pred=jnp.all(eigvals > 0),
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )

        vec_constraints = jnp.array(
            [
                valid_constraints_vec_delta_size,
                valid_constraints_vec_delta_range_lo,
                valid_constraints_vec_delta_range_hi,
                valid_constraints_vec_delta_sum,
                valid_constraints_mat_Qbar_psd,
            ]
        )
        valid_constraints = jax.lax.select(
            pred=jnp.all(vec_constraints),
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )
        # return valid_constraints
        # HACK::
        return True



@chex.dataclass
class ParamsDcc:
    """
    Collect all the parameters of a DCC-GARCH model.
    """

    uvar_vol: ParamsUVarVol
    mvar_cor: ParamsMVarCor


@chex.dataclass
class ParamsModel:
    """
    Generic parent class for holding the parameters of an entire DCC-X-GARCH
    model, where X depends on the choice of specifications of the
    innovation process z_t.
    """

    mean: ParamsMean
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


@chex.dataclass
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


def _calc_normalized_unexpected_excess_rtn(
    vec_sigma: jpt.Float[jpt.Array, "dim"], vec_epsilon: jpt.Float[jpt.Array, "dim"]
) -> jpt.Float[jpt.Array, "dim"]:
    """
    Return u_t = D_t^{-1} \\epsilon_t
    """
    mat_inv_D = jnp.diag(1 / vec_sigma)
    vec_u = mat_inv_D @ vec_epsilon

    return vec_u


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


def _calc_mat_Gamma(
    mat_Q: jpt.Float[jpt.Array, "dim dim"]
) -> jpt.Float[jpt.Array, "dim dim"]:
    """
    Return the \\Gamma_t matrix of the DCC model
    """
    mat_Qstar_inv_sqrt = jnp.diag(jnp.diag(mat_Q) ** (-1 / 2))
    mat_Gamma = mat_Qstar_inv_sqrt @ mat_Q @ mat_Qstar_inv_sqrt

    return mat_Gamma


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

    _func = vmap(sgt._time_varying_lbda_params, in_axes=[1, 0, 0])
    def _body_fun(tt, mat_lbda):
        vec_lbda_t = _func(mat_lbda_tvparams, mat_lbda[tt - 1], mat_z[tt - 1, :])
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

    _func = vmap(sgt._time_varying_pq_params, in_axes=[1, 0, 0])
    def _body_fun(tt, mat_pq):
        vec_pq_t = _func(mat_pq_tvparams, mat_pq[tt - 1], mat_z[tt - 1, :])
        mat_pq = mat_pq.at[tt].set(vec_pq_t)
        return mat_pq

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
    mat_q0_tvparams = params_dcc_sgt_garch.sgt.mat_q0_tvparams

    vec_mu = params_dcc_sgt_garch.mean.vec_mu

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
    logger.debug("Begin _calc_trajectory_uvar_vol")
    mat_sigma = _calc_trajectory_uvar_vol(
        mat_epsilon=mat_epsilon,
        vec_sigma_init_t0=vec_sigma_init_t0,
        vec_omega=vec_omega,
        vec_beta=vec_beta,
        vec_alpha=vec_alpha,
        vec_psi=vec_psi,
    )
    logger.debug("End _calc_trajectory_uvar_vol")

    # Calculate the multivariate covariance \Sigma_t
    logger.debug("Begin _calc_trajectory_mvar_cov")
    tns_Sigma = _calc_trajectory_mvar_cov(
        mat_epsilon=mat_epsilon,
        mat_sigma=mat_sigma,
        mat_Q_0=mat_Q_init_t0,
        vec_delta=vec_delta,
        mat_Qbar=mat_Qbar,
    )
    logger.debug("End _calc_trajectory_mvar_cov")

    # Calculate the innovations z_t = \Sigma_t^{-1/2} \epsilon_t
    logger.debug("Begin _calc_trajectory_innovations")
    mat_z = _calc_trajectory_innovations(mat_epsilon=mat_epsilon, tns_Sigma=tns_Sigma)
    logger.debug("End _calc_trajectory_innovations")

    # Calculate the normalized unexpected returns u_t
    logger.debug("Begin _calc_trajectory_normalized_unexp_returns")
    mat_u = _calc_trajectory_normalized_unexp_returns(
        mat_sigma=mat_sigma, mat_epsilon=mat_epsilon
    )
    logger.debug("End _calc_trajectory_normalized_unexp_returns")

    # Calculate the time-varying parameters \lambda_t
    # associated with the innovations z_t
    logger.debug("Begin _calc_trajectory_innovations_*")
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
    logger.debug("End _calc_trajectory_innovations_*")

    return mat_lbda, mat_p0, mat_q0, mat_epsilon, mat_sigma, tns_Sigma, mat_z, mat_u


def simulate_dcc_sgt_garch(
    key: KeyArrayLike,
    dim: int,
    num_sample: int,
    params_dcc_sgt_garch: ParamsDccSgtGarch,
    inittimecond_dcc_sgt_garch: InitTimeConditionDccSgtGarch,
    data_simreturns_savepath: os.PathLike,
) -> SimulatedReturns:
    """
    Simulate DCC-SGT-GARCH.
    """
    # Get the simulated innovations z_t
    siminnov = sgt.sample_mvar_timevarying_sgt(
        key=key,
        num_sample=num_sample,
        params_z_sgt_true=params_dcc_sgt_garch.sgt,
        inittimecond_z_sgt=inittimecond_dcc_sgt_garch.sgt,
        save_path=None,
    )

    # Obtain the simulated asset returns R_t
    simreturns = _simulate_returns(
        dim=dim,
        num_sample=num_sample,
        siminnov=siminnov,
        params_mean_true=params_dcc_sgt_garch.mean,
        params_dcc_true=params_dcc_sgt_garch.dcc,
        inittimecond_dcc=inittimecond_dcc_sgt_garch.dcc,
        data_simreturns_savepath=data_simreturns_savepath,
    )

    return simreturns


def _simulate_returns(
    dim: int,
    num_sample: int,
    siminnov: sgt.SimulatedInnovations,
    params_mean_true: ParamsMean,
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
        num_sample=num_sample, dim=dim, vec_mu=params_mean_true.vec_mu
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
) -> jpt.Float:
    """
    (Negative) of the likelihood of the
    DCC-time-varying-SGT-Asymmetric-GARCH(1,1) model
    """
    neg_loglik: bool = True
    normalize_loglik : bool = True

    num_sample = mat_returns.shape[0]

    logger.debug("Begin calc_trajectories")
    mat_lbda, mat_p0, mat_q0, _, _, tns_Sigma, mat_z, _ = calc_trajectories(
        mat_returns=mat_returns,
        params_dcc_sgt_garch=params_dcc_sgt_garch,
        inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch,
    )
    logger.debug("End calc_trajectories")

    # Compute {\log\det \Sigma_t}
    logger.debug("Begin \\log\\det\\Sigma_t")
    _, vec_logdet_Sigma = jnp.linalg.slogdet(tns_Sigma)
    logger.debug("End \\log\\det\\Sigma_t")

    # Compute the log-likelihood of g(z_t | \theta_t)
    # where g \sim SGT
    logger.debug("Begin log-likelihood of z_t")
    sgt_loglik = sgt.loglik_mvar_timevarying_sgt(
        data=mat_z,
        mat_lbda=mat_lbda,
        mat_p0=mat_p0,
        mat_q0=mat_q0,
        neg_loglik=False,
    )
    logger.debug("End log-likelihood of z_t")

    # Objective function of DCC model
    loglik = sgt_loglik - 0.5 * vec_logdet_Sigma.sum()

    if normalize_loglik:
        loglik = loglik / num_sample

    if neg_loglik:
        loglik = -1 * loglik

    return loglik


def params_to_arr(
    params_dataclass: sgt.ParamsZSgt | ParamsMean | ParamsUVarVol | ParamsMVarCor,
) -> jpt.Array:
    """
    Take a parameter dataclass and flatten the fields to an array
    """
    # Note special treatment for when params_dataclass is
    # dcc.ParamsMVarCor
    dict_params = dataclasses.asdict(params_dataclass)

    lst = []
    # Sort the keys for consistency
    for key in sorted(dict_params.keys()):
        lst.append(dict_params[key])

    arr = jnp.concatenate(lst)

    # Special treament for ParamsZSgt
    if isinstance(params_dataclass, sgt.ParamsZSgt):
        arr = arr.flatten()

    return arr


def _make_params_from_arr_z_sgt(x : jpt.Array, dim : int) -> sgt.ParamsZSgt:
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

    # NOTE: Must be organized in alphabetical order
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


def _make_params_from_arr_mean(x : jpt.Array, dim : int) -> ParamsMean:
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


def _make_params_from_arr_dcc_uvar_vol(x : jpt.Array, dim :int) -> ParamsUVarVol:
    """
    Take a vector x and split them into parameters related to the
    univariate GARCH processes.
    """
    # NOTE: Must organize in alphabetical order
    params_uvar_vol = ParamsUVarVol(
        vec_alpha=x[0:dim],
        vec_beta=x[dim : 2 * dim],
        vec_omega=x[2 * dim : 3 * dim],
        vec_psi=x[3 * dim :],
    )
    return params_uvar_vol


def _make_params_from_arr_dcc_mvar_cor(
    x : jpt.Array,
    dim : int,
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

def params_update(params : sgt.ParamsZSgt | ParamsMean | ParamsUVarVol | ParamsMVarCor, 
                  params_dcc_sgt_garch : ParamsDccSgtGarch) -> ParamsDccSgtGarch:
    """
    Convenient function to update parameters into the main parameters dataclass.
    """
    if isinstance(params, ParamsZSgt):
        params_dcc_sgt_garch.sgt = params
    elif isinstance(params, ParamsMean):
        params_dcc_sgt_garch.mean = params
    elif isinstance(params, ParamsUVarVol):
        params_dcc_sgt_garch.dcc.uvar_vol = params
    elif isinstance(params, ParamsMVarCor):
        params_dcc_sgt_garch.dcc.mvar_cor = params
    else:
        raise ValueError("Incorrect specification")

    return params_dcc_sgt_garch


def _objfun_dcc_loglik(
    x : jpt.Array,
    make_params_from_arr,
    params_dcc_sgt_garch: ParamsDccSgtGarch,
    inittimecond_dcc_sgt_garch: InitTimeConditionDccSgtGarch,
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"],
) -> tp.Tuple[jpt.Float, ParamsDccSgtGarch]:
    dim = mat_returns.shape[1]

    # Construct the dict for the parameters
    # that are to be optimized over
    optimizing_params_name = make_params_from_arr.__name__
    if optimizing_params_name == "__make_params_from_arr_dcc_mvar_cor":
        # Special treatment in handling the estimate of
        # \hat{\bar{Q}} (i.e. volatility-targetting estimation method)
        _params = make_params_from_arr(x, dim, params_dcc_sgt_garch)
    else:
        _params = make_params_from_arr(x, dim)

    # # HACK:
    # # Check if constraints on the parameters are violated
    # valid_constraints = _params.check_constraints()
    # if valid_constraints is False:
    #     neg_loglik = jnp.inf
    #     return neg_loglik
    #

    params_dcc_sgt_garch = params_update(params = _params, params_dcc_sgt_garch=params_dcc_sgt_garch)


    # Compute (negative) of the log-likelihood
    # try:
    #     neg_loglik = dcc_sgt_loglik(
    #         mat_returns=mat_returns,
    #         params_dcc_sgt_garch=params_dcc_sgt_garch,
    #         inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch,
    #     )
    # except FloatingPointError as e:
    #     logger.info(str(e))
    #     logger.info(f"{params_dcc_sgt_garch}")
    #     neg_loglik = jnp.inf
    # except Exception as e:
    #     logger.info(f"{e}")
    #     neg_loglik = jnp.inf
    #

    logger.debug(params_dcc_sgt_garch)
    neg_loglik = dcc_sgt_loglik(
        mat_returns=mat_returns,
        params_dcc_sgt_garch=params_dcc_sgt_garch,
        inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch,
    )

    return neg_loglik, params_dcc_sgt_garch


# def _make_params_array_from_dict_params(
#     dict_params: tp.Dict[str, jpt.Array]
# ) -> jpt.Array:
#     """
#     Take in a dictionary of parameters and flatten them to an
#     array in preparation for optimization.
#     """
#     x0 = itertools.chain.from_iterable(dict_params.values())
#     x0 = list(x0)
#     x0 = np.array(x0)
#     x0 = jnp.array(x0)
#
#     return x0

def _projection_unit_simplex(x: jnp.ndarray) -> jnp.ndarray:
  """Projection onto the unit simplex."""
  s = 1.0
  n_features = x.shape[0]
  u = jnp.sort(x)[::-1]
  cumsum_u = jnp.cumsum(u)
  ind = jnp.arange(n_features) + 1
  cond = s / ind + (u - cumsum_u / ind) > 0
  idx = jnp.count_nonzero(cond)
  return jax.nn.relu(s / idx + (x - cumsum_u[idx - 1] / idx))

def projection_simplex(x: jnp.ndarray, value: float = 1.0) -> jnp.ndarray:
  r"""Projection onto a simplex:

  .. math::

    \underset{p}{\text{argmin}} ~ ||x - p||_2^2 \quad \textrm{subject to} \quad
    p \ge 0, p^\top 1 = \text{value}

  By default, the projection is onto the probability simplex.

  Args:
    x: vector to project, an array of shape (n,).
    value: value p should sum to (default: 1.0).
  Returns:
    projected vector, an array with the same shape as ``x``.
  """
  if value is None:
    value = 1.0
  return value * _projection_unit_simplex(x / value)

def projection_l1_sphere(x: jnp.ndarray, value: float = 1.0) -> jnp.ndarray:
  r"""Projection onto the l1 sphere:

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    ||y||_1 = \text{value}

  Args:
    x: array to project.
    value: radius of the sphere.

  Returns:
    output array, with the same shape as ``x``.
  """
  return jnp.sign(x) * projection_simplex(jnp.abs(x), value)

def projection_l1_ball(x: jnp.ndarray, max_value: float = 1.0) -> jnp.ndarray:
  r"""Projection onto the l1 ball:

  .. math::

    \underset{y}{\text{argmin}} ~ ||x - y||_2^2 \quad \textrm{subject to} \quad
    ||y||_1 \le \text{max_value}

  Args:
    x: array to project.
    max_value: radius of the ball.

  Returns:
    output array, with the same structure as ``x``.
  """
  l1_norm = jax.numpy.linalg.norm(x, ord=1)
  return jax.lax.cond(l1_norm <= max_value,
                      lambda _: x,
                      lambda _: projection_l1_sphere(x, max_value),
                      operand=None)


def build_estimation_step(optimizer: optax.GradientTransformation, loss_fn : tp.Callable[[jpt.Array, ParamsDccSgtGarch], tp.Tuple[jpt.Float, ParamsDccSgtGarch]]):
    """
    Builds a function for executing a single step in the optimization
    """

    @jit
    def _update(x, opt_state, params_dcc_sgt_garch):
        grads, params_dcc_sgt_garch = jax.grad(loss_fn, has_aux=True)(x, params_dcc_sgt_garch)
        updates, opt_state = optimizer.update(grads, opt_state)
        x = optax.apply_updates(x, updates)

        return x, opt_state, params_dcc_sgt_garch

    return _update

def fit(optimizer : optax.GradientTransformation, 
        loss_fn : tp.Callable[[jpt.Array, ParamsDccSgtGarch], tp.Tuple[jpt.Float, ParamsDccSgtGarch]], 
        x_init : jpt.Array, 
        params_dcc_sgt_garch : ParamsDccSgtGarch, 
        make_params_from_arr,
        dim : int,
        numiter : int,
        lst_projection : None | tp.List[tp.Callable[[jpt.PyTree], jpt.PyTree]] = None
        ) -> ParamsDccSgtGarch:
    """
    Fit
    """
    train_step = build_estimation_step(optimizer, loss_fn)

    x = x_init
    opt_state = optimizer.init(x)

    for _ in range(numiter):
        x, opt_state, params_dcc_sgt_garch = train_step(x, opt_state, params_dcc_sgt_garch)

        if lst_projection is not None:
            for projection in lst_projection:
                x = projection(x)

        # Special treatment in handling the estimate of
        # \hat{\bar{Q}} (i.e. volatility-targetting estimation method)
        if make_params_from_arr.__name__ == "__make_params_from_arr_dcc_mvar_cor":
            _params = make_params_from_arr(x, dim, params_dcc_sgt_garch)
            params_dcc_sgt_garch = params_update(_params, params_dcc_sgt_garch=params_dcc_sgt_garch)


        logger.debug(f"Value function at {loss_fn(x, params_dcc_sgt_garch)}")

    return params_dcc_sgt_garch


def dcc_sgt_garch_optimization(
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"],
    params_dcc_sgt_garch: ParamsDccSgtGarch,
    inittimecond_dcc_sgt_garch: InitTimeConditionDccSgtGarch,
    verbose: bool = False,
    grand_maxiter: int = 5,
    inner_maxiter: int = 100,
    tol: float = 1e-3,
    jit: bool = True,
    solver_z=jaxopt.NonlinearCG,
    solver_mean=jaxopt.NonlinearCG,
    solver_dcc_uvar_vol=jaxopt.ProjectedGradient,
    solver_dcc_mvar_cor=jaxopt.NonlinearCG,
):
    """
    Run DCC-SGT-GARCH optimization
    """
    logger.info(f"Begin DCC-SGT-GARCH optimization.")

    dim = mat_returns.shape[1]

    #################################################################
    ## Setup partial objective functions
    #################################################################
    objfun_dcc_loglik_opt_params_z : tp.Callable[[jpt.Array, ParamsDccSgtGarch], tp.Tuple[jpt.Float, ParamsDccSgtGarch]]= lambda x, params_dcc_sgt_garch: _objfun_dcc_loglik(
        x=x,
        make_params_from_arr=_make_params_from_arr_z_sgt,
        params_dcc_sgt_garch=params_dcc_sgt_garch,
        inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch,
        mat_returns=mat_returns,
    )

    objfun_dcc_loglik_opt_params_mean : tp.Callable[[jpt.Array, ParamsDccSgtGarch], tp.Tuple[jpt.Float, ParamsDccSgtGarch]] = ( lambda x, params_dcc_sgt_garch: _objfun_dcc_loglik(
            x=x,
            make_params_from_arr=_make_params_from_arr_mean,
            params_dcc_sgt_garch=params_dcc_sgt_garch,
            inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch,
            mat_returns=mat_returns,
        )
    )

    objfun_dcc_loglik_opt_params_dcc_uvar_vol : tp.Callable[[jpt.Array, ParamsDccSgtGarch], tp.Tuple[jpt.Float, ParamsDccSgtGarch]]= (
        lambda x, params_dcc_sgt_garch: _objfun_dcc_loglik(
            x=x,
            make_params_from_arr=_make_params_from_arr_dcc_uvar_vol,
            params_dcc_sgt_garch=params_dcc_sgt_garch,
            inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch,
            mat_returns=mat_returns,
        )
    )


    # Note special treatmeant here for updating DCC Q_t 
    # parameters
    def __make_params_from_arr_dcc_mvar_cor(x : jpt.Array, dim : int, params_dcc_sgt_garch : ParamsDccSgtGarch): 
        return _make_params_from_arr_dcc_mvar_cor(x = x,
                                                  dim = dim,
                                                  mat_returns=mat_returns, 
                                                  params_dcc_sgt_garch=params_dcc_sgt_garch, 
                                                  inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch)

    objfun_dcc_loglik_opt_params_dcc_mvar_cor = (
        lambda x, params_dcc_sgt_garch: _objfun_dcc_loglik(
            x=x,
            make_params_from_arr=__make_params_from_arr_dcc_mvar_cor,
            params_dcc_sgt_garch=params_dcc_sgt_garch,
            inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch,
            mat_returns=mat_returns,
        )
    )


    #################################################################
    ## Setup partial solvers
    #################################################################
    # solver_obj_z = solver_z(
    #     fun=objfun_dcc_loglik_opt_params_z,
    #     verbose=verbose,
    #     maxiter=maxiter,
    #     tol=tol,
    #     jit = jit
    # )
    #
    # solver_obj_mean = solver_mean(
    #     fun=objfun_dcc_loglik_opt_params_mean,
    #     verbose=verbose,
    #     maxiter=maxiter,
    #     tol=tol,
    #     jit = jit
    # )
    #
    # solver_obj_dcc_uvar_vol = solver_dcc_uvar_vol(
    #     fun=objfun_dcc_loglik_opt_params_dcc_uvar_vol,
    #     verbose=verbose,
    #     maxiter=maxiter,
    #     tol=tol,
    #     projection = jaxopt.projection.projection_non_negative,
    #     jit = jit
    # )
    #
    # solver_obj_dcc_mvar_cor = solver_dcc_mvar_cor(
    #     fun=objfun_dcc_loglik_opt_params_dcc_mvar_cor,
    #     verbose=verbose,
    #     maxiter=maxiter,
    #     tol=tol,
    #     jit = jit
    # )
    
    learning_rate_schedule = optax.piecewise_constant_schedule(init_value = 1.0,
                                                               boundaries_and_scales= {0: 1e-4, 1: 1e-1}
                                                               )

    start_learning_rate = 1e-1


    projection_strictly_positive = lambda x : optax.projections.projection_box(x, lower = 1e-3, upper = jnp.inf)
    projection_hypercube = lambda x : optax.projections.projection_box(x, lower = 0, upper = 1)

    learning_schedule = optax.exponential_decay(init_value = start_learning_rate, transition_steps=grand_maxiter, decay_rate=1e-1)


    #################################################################
    ## Iterative log-likelihood optimization
    #################################################################
    neg_loglik_optval = None
    for iter in range(grand_maxiter):
        # Init optimizers
        learning_rate = float(learning_schedule(iter))
        optimizer_z = optax.adam(learning_rate)
        optimizer_mean = optax.adam(learning_rate)
        optimizer_dcc_uvar_vol = optax.adam(learning_rate)
        optimizer_dcc_mvar_cor = optax.adam(learning_rate)

        #################################################################
        ## Step 1: Optimize for the parameters of z_t
        #################################################################
        x0_z = params_to_arr(params_dcc_sgt_garch.sgt)
        neg_loglik_val_beg = dcc_sgt_loglik(mat_returns=mat_returns,
                                        params_dcc_sgt_garch=params_dcc_sgt_garch,
                                        inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch)
        params_dcc_sgt_garch = fit(optimizer=optimizer_z,
                                   loss_fn=objfun_dcc_loglik_opt_params_z,
                                   x_init = x0_z,
                                   params_dcc_sgt_garch=params_dcc_sgt_garch,
                                   make_params_from_arr=_make_params_from_arr_z_sgt,
                                   dim = dim,
                                   numiter = inner_maxiter)
        neg_loglik_val_end = dcc_sgt_loglik(mat_returns=mat_returns,
                                        params_dcc_sgt_garch=params_dcc_sgt_garch,
                                        inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch)
        logger.info(f"..... {iter}/{grand_maxiter}: Step 1/4 --- Optimize innovation z_t params. Objective function value changed from {neg_loglik_val_beg} to {neg_loglik_val_end}.")


        #################################################################
        ## Step 2: Optimize for the parameters of the mean \mu
        #################################################################
        x0_mean = params_to_arr(params_dcc_sgt_garch.mean)
        neg_loglik_val_beg = dcc_sgt_loglik(mat_returns=mat_returns,
                                        params_dcc_sgt_garch=params_dcc_sgt_garch,
                                        inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch)
        params_dcc_sgt_garch = fit(optimizer=optimizer_mean,
                                   loss_fn=objfun_dcc_loglik_opt_params_mean,
                                   x_init = x0_mean,
                                   params_dcc_sgt_garch=params_dcc_sgt_garch,
                                   make_params_from_arr=_make_params_from_arr_mean,
                                   dim = dim,
                                   numiter = inner_maxiter)
        neg_loglik_val_end = dcc_sgt_loglik(mat_returns=mat_returns,
                                        params_dcc_sgt_garch=params_dcc_sgt_garch,
                                        inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch)
        logger.info(f"..... {iter}/{grand_maxiter}: Step 2/4 --- Optimize mean \\mu params. Objective function value changed from {neg_loglik_val_beg} to {neg_loglik_val_end}.")


        #################################################################
        ## Step 3: Optimize for the parameters of the univariate vol's
        ##         \sigma_{i,t}'s
        #################################################################
        x0_dcc_uvar_vol = params_to_arr(params_dcc_sgt_garch.dcc.uvar_vol)
        neg_loglik_val_beg = dcc_sgt_loglik(mat_returns=mat_returns,
                                            params_dcc_sgt_garch=params_dcc_sgt_garch,
                                            inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch)
        params_dcc_sgt_garch = fit(optimizer=optimizer_dcc_uvar_vol,
                                   loss_fn=objfun_dcc_loglik_opt_params_dcc_uvar_vol,
                                   x_init = x0_dcc_uvar_vol,
                                   params_dcc_sgt_garch=params_dcc_sgt_garch,
                                   make_params_from_arr=_make_params_from_arr_dcc_uvar_vol,
                                   dim = dim,
                                   numiter = inner_maxiter,
                                   lst_projection=[projection_strictly_positive])
        neg_loglik_val_end = dcc_sgt_loglik(mat_returns=mat_returns,
                                            params_dcc_sgt_garch=params_dcc_sgt_garch,
                                            inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch)
        logger.info(f"..... {iter}/{grand_maxiter}: Step 3/4 --- Optimize univariate vol \\sigma params. Objective function value changed from {neg_loglik_val_beg} to {neg_loglik_val_end}.")


        #################################################################
        ## Step 4: Optimize for the parameters of the multivariate DCC
        ##         Q_t's
        #################################################################
        # Note special treatment for the initial guess for the
        # DCC multivariate parameters
        x0_dcc_mvar_cor = params_dcc_sgt_garch.dcc.mvar_cor.vec_delta
        neg_loglik_val_beg = dcc_sgt_loglik(mat_returns=mat_returns,
                                            params_dcc_sgt_garch=params_dcc_sgt_garch,
                                            inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch)
        params_dcc_sgt_garch = fit(optimizer=optimizer_dcc_mvar_cor,
                                   loss_fn=objfun_dcc_loglik_opt_params_dcc_mvar_cor,
                                   x_init = x0_dcc_mvar_cor,
                                   params_dcc_sgt_garch=params_dcc_sgt_garch,
                                   make_params_from_arr= __make_params_from_arr_dcc_mvar_cor,
                                   dim = dim,
                                   numiter = inner_maxiter,
                                   lst_projection=[projection_hypercube, projection_l1_ball])
        neg_loglik_val_end = dcc_sgt_loglik(mat_returns=mat_returns,
                                            params_dcc_sgt_garch=params_dcc_sgt_garch,
                                            inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch)
        logger.info(f"..... {iter}/{grand_maxiter}: Step 4/4 --- Optimize multivariate DCC Q_t params. Objective function value changed from {neg_loglik_val_beg} to {neg_loglik_val_end}.")



    # Record the value of the last negative log-likelihood
    neg_loglik_optval = dcc_sgt_loglik(mat_returns=mat_returns,
                                       params_dcc_sgt_garch=params_dcc_sgt_garch,
                                       inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch)

    logger.info(f"Done DCC-GARCH optimization.")
    return neg_loglik_optval, params_dcc_sgt_garch


if __name__ == "__main__":
    pass
