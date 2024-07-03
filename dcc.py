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

import uuid


import dataclasses
from dataclasses import dataclass

from datetime import datetime

import utils


current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger(__name__)
logging.basicConfig(
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(f"logs/{current_time}_dcc.log", mode="w"),
        logging.StreamHandler(),
    ],
)


import typing as tp

import optax

import numpy as np

import innovations
from innovations import (
    ParamsZSgt,
    SimulatedInnovations,
)

# HACK:
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod
# jax.config.update("jax_debug_nans", True)  # Should disable in full prod
# jax.config.update("jax_disable_jit", True)  # Should disable in full prod


@chex.dataclass
class ParamsDcc:
    """
    Placeholder for all params in a DCC model
    """

    pass


@chex.dataclass
class ParamsMean(ParamsDcc):
    """
    Parameters for the mean returns in a DCC-GARCH model
    """

    vec_mu: jpt.Float[jpt.Array, "dim"]

    def check_constraints(self) -> bool:
        # No particular constraints on the mean
        valid = True
        return valid


@chex.dataclass
class ParamsUVarVol(ParamsDcc):
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
        return valid_constraints


@dataclass
class ParamsMVarCor(ParamsDcc):
    """
    Parameters for the multivariate Q_t process in a DCC-GARCH model
    """

    vec_delta: jpt.Float[jpt.Array, "2"]


@dataclass
class ParamsMVarCorQbar(ParamsDcc):
    """
    Convenient to separate out the treatment for \\bar{Q}
    as this will not be optimized by MLE
    """

    mat_Qbar: jpt.Float[jpt.Array, "dim dim"]


@dataclass
class ModelDcc:
    """
    Generic parent class for holding the parameters of an entire DCC-X-GARCH
    model, where X depends on the choice of specifications of the
    innovation process z_t.
    """

    mean: ParamsMean
    uvar_vol: ParamsUVarVol
    mvar_cor: ParamsMVarCor
    mvar_corQbar: ParamsMVarCorQbar


@dataclass
class ModelDccGaussianGarch(ModelDcc):
    """
    DCC-Gaussian-GARCH model
    """

    innov_z: innovations.ParamsZGaussian


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
class InitTimeConditionDccGaussianGarch(InitTimeConditionDcc):
    dcc: InitTimeConditionDcc


@chex.dataclass
class SimulatedReturns:
    """
    Data class for keeping track of the parameters and data
    of simulated returns.
    """

    # Hyperparams for record keeping
    hashid: str
    seed: int

    # Dimension (i.e. number of assets) and number of time samples
    dim: int
    num_sample: int

    ################################################################
    ## Object for the innovations z_t
    ################################################################
    siminnov: SimulatedInnovations

    ################################################################
    ## Parameters
    ################################################################
    model_dcc_true: ModelDcc

    ################################################################
    ## Simulated data
    ################################################################
    # Unexpected excess returns \epsilon_t
    data_mat_epsilon: jpt.Array

    # Univariate volatilities \sigma_{i,t}'s
    data_mat_sigma: jpt.Array

    # Normalized unexpected returns u_t
    data_mat_u: jpt.Array

    # DCC parameters Q_t
    data_tns_Q: jpt.Array

    # DCC covariances \Sigma_t
    data_tns_Sigma: jpt.Array

    # Simulated asset returns
    data_mat_returns: jpt.Array


@chex.dataclass
class EstimationResults:
    """
    Data class for storing the estimation results.
    """

    # If True, then the estimation is good. If False,
    # then the estimation is bad (e.g. NaN)
    valid_optimization: jpt.Bool

    # Value of the negative of the log-likelihood.
    neg_loglik_val: jpt.Float

    # Estimated parameters of the DCC-distr-GARCH model
    dcc_model: ModelDcc


# Convenient typing
TypeParams = tp.TypeVar(
    "TypeParams",
    innovations.ParamsZGaussian,
    ParamsMean,
    ParamsUVarVol,
    ParamsMVarCor,
)
TypeModelDcc = tp.TypeVar("TypeModelDcc", ModelDcc, ModelDccGaussianGarch)
TypeInitTimeConditionDcc = tp.TypeVar(
    "TypeInitTimeConditionDcc",
    InitTimeConditionDcc,
    InitTimeConditionDccGaussianGarch,
)


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
    model_dcc: TypeModelDcc,
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
    num_sample = jnp.shape(data_z)[0]
    dim = jnp.shape(data_z)[1]

    vec_omega = model_dcc.uvar_vol.vec_omega
    vec_beta = model_dcc.uvar_vol.vec_beta
    vec_alpha = model_dcc.uvar_vol.vec_alpha
    vec_psi = model_dcc.uvar_vol.vec_psi

    vec_delta = model_dcc.mvar_cor.vec_delta
    mat_Qbar = model_dcc.mvar_corQbar.mat_Qbar

    mat_Sigma_init_t0 = inittimecond_dcc.mat_Sigma_init_t0
    mat_Q_init_t0 = inittimecond_dcc.mat_Q_init_t0

    logger.debug(f"Begin DCC simulation on num_sample = {num_sample} and dim = {dim}")

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


@jit
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

    def _func(x, y):
        vec_sigma2_t = _calc_asymmetric_garch_sigma2(
            vec_sigma_t_minus_1=x,
            vec_epsilon_t_minus_1=y,
            vec_omega=vec_omega,
            vec_beta=vec_beta,
            vec_alpha=vec_alpha,
            vec_psi=vec_psi,
        )
        vec_sigma_t = jnp.sqrt(vec_sigma2_t)
        return vec_sigma_t, vec_sigma_t

    _, mat_sigma = jax.lax.scan(_func, init=vec_sigma_init_t0, xs=mat_epsilon)
    mat_sigma = jnp.insert(arr=mat_sigma, obj=0, values=vec_sigma_init_t0, axis=0)
    mat_sigma = jnp.delete(mat_sigma, -1, axis=0)

    return mat_sigma


@jit
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
    vec_u_0 = _calc_normalized_unexpected_excess_rtn(
        vec_sigma=mat_sigma[0, :], vec_epsilon=mat_epsilon[0, :]
    )
    mat_Gamma_0 = _calc_mat_Gamma(mat_Q=mat_Q_0)
    mat_Sigma_0 = _calc_mat_Sigma(vec_sigma=mat_sigma[0, :], mat_Gamma=mat_Gamma_0)

    def _func(carry, xs):
        vec_sigma_t, vec_epsilon_t = xs
        vec_u_t_minus_1, mat_Q_t_minus_1, _ = carry

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

        # Compute u_t
        vec_u_t = _calc_normalized_unexpected_excess_rtn(
            vec_sigma=vec_sigma_t, vec_epsilon=vec_epsilon_t
        )

        carry = (vec_u_t, mat_Q_t, mat_Sigma_t)
        return carry, carry

    _, carry = jax.lax.scan(
        _func,
        init=(vec_u_0, mat_Q_0, mat_Sigma_0),
        xs=(mat_sigma[1:, :], mat_epsilon[1:, :]),
    )
    # mat_u = carry[0]
    # tns_Q = carry[1]
    tns_Sigma = carry[2]

    # mat_u = jnp.insert(mat_u, 0, vec_u_0, axis = 0)
    # tns_Q = jnp.insert(tns_Q, 0, mat_Q_0, axis = 0)
    tns_Sigma = jnp.insert(tns_Sigma, 0, mat_Sigma_0, axis=0)

    return tns_Sigma


@jit
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


@jit
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


def calc_general_trajectories(
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"],
    model_dcc: TypeModelDcc,
    inittimecond_dcc_distr_garch: TypeInitTimeConditionDcc,
) -> tp.Tuple[
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_epsilon
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_sigma
    jpt.Float[jpt.Array, "num_sample dim dim"],  # tns_Sigma
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_z
    jpt.Float[jpt.Array, "num_sample dim"],  # mat_u
]:
    """
    Given parameters, return the trajectories
    {\\epsilon_t}, {\\sigma_{i,t}}, {\\Sigma_t}, {z_t}, {u_t}
    """
    # Compute \epsilon_t = R_t - \mu_t
    mat_epsilon = _calc_demean_returns(
        mat_returns=mat_returns, vec_mu=model_dcc.mean.vec_mu
    )

    # Calculate the univariate vol's \sigma_{i,t}'s
    logger.debug("Begin _calc_trajectory_uvar_vol")
    mat_sigma = _calc_trajectory_uvar_vol(
        mat_epsilon=mat_epsilon,
        vec_sigma_init_t0=inittimecond_dcc_distr_garch.vec_sigma_init_t0,
        vec_omega=model_dcc.uvar_vol.vec_omega,
        vec_beta=model_dcc.uvar_vol.vec_beta,
        vec_alpha=model_dcc.uvar_vol.vec_alpha,
        vec_psi=model_dcc.uvar_vol.vec_psi,
    )
    logger.debug("End _calc_trajectory_uvar_vol")

    # Calculate the multivariate covariance \Sigma_t
    logger.debug("Begin _calc_trajectory_mvar_cov")
    tns_Sigma = _calc_trajectory_mvar_cov(
        mat_epsilon=mat_epsilon,
        mat_sigma=mat_sigma,
        mat_Q_0=inittimecond_dcc_distr_garch.mat_Q_init_t0,
        vec_delta=model_dcc.mvar_cor.vec_delta,
        mat_Qbar=model_dcc.mvar_corQbar.mat_Qbar,
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

    return mat_epsilon, mat_sigma, tns_Sigma, mat_z, mat_u


def _simulate_returns(
    seed: int,
    hashid: str,
    dim: int,
    num_sample: int,
    siminnov: SimulatedInnovations,
    model_dcc_true: TypeModelDcc,
    inittimecond_dcc: TypeInitTimeConditionDcc,
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
            model_dcc=model_dcc_true,
            inittimecond_dcc=inittimecond_dcc,
        )
    )

    # Set the asset mean
    vec_mu = model_dcc_true.mean.vec_mu
    mat_mu = _calc_mean_return(num_sample=num_sample, dim=dim, vec_mu=vec_mu)

    # Asset returns
    data_mat_returns = mat_mu + data_mat_epsilon

    logger.debug("Done simulating returns")

    # Save
    simreturns = SimulatedReturns(
        hashid=hashid,
        seed=seed,
        dim=dim,
        num_sample=num_sample,
        siminnov=siminnov,
        model_dcc_true=model_dcc_true,
        data_mat_epsilon=data_mat_epsilon,
        data_mat_sigma=data_mat_sigma,
        data_mat_u=data_mat_u,
        data_tns_Q=data_tns_Q,
        data_tns_Sigma=data_tns_Sigma,
        data_mat_returns=data_mat_returns,
    )

    return simreturns


def simulate_dcc_gaussian_garch(
    seed: int,
    hashid: str,
    key: KeyArrayLike,
    dim: int,
    num_sample: int,
    model_dcc_gaussian_garch: ModelDccGaussianGarch,
    inittimecond_dcc_gaussian_garch: InitTimeConditionDccGaussianGarch,
) -> SimulatedReturns:
    """
    Simulate DCC-Gaussian-GARCH
    """
    # Get the simulated innovations z_t
    siminnov = innovations.sample_mvar_gaussian(
        key=key,
        num_sample=num_sample,
        params_z_gaussian_true=model_dcc_gaussian_garch.innov_z,
    )

    # Obtain the simulated asset returns R_t
    simreturns = _simulate_returns(
        seed=seed,
        hashid=hashid,
        dim=dim,
        num_sample=num_sample,
        siminnov=siminnov,
        model_dcc_true=model_dcc_gaussian_garch,
        inittimecond_dcc=inittimecond_dcc_gaussian_garch,
    )

    return simreturns


def dcc_gaussian_loglik(
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"],
    model_dcc_gaussian_garch: ModelDccGaussianGarch,
    inittimecond_dcc_gaussian_garch: InitTimeConditionDccGaussianGarch,
) -> jpt.Float:
    """
    (Negative) of the likelihood of the
    DCC-Gaussian-Asymmetric-GARCH(1,1) model
    """
    neg_loglik: bool = True
    normalize_loglik: bool = True

    num_sample = mat_returns.shape[0]

    logger.debug("Begin calc_trajectories")
    _, _, tns_Sigma, mat_z, _ = calc_general_trajectories(
        mat_returns=mat_returns,
        model_dcc=model_dcc_gaussian_garch,
        inittimecond_dcc_distr_garch=inittimecond_dcc_gaussian_garch,
    )
    logger.debug("End calc_trajectories")

    # Compute {\log\det \Sigma_t}
    logger.debug("Begin \\log\\det\\Sigma_t")
    _, vec_logdet_Sigma = jnp.linalg.slogdet(tns_Sigma)
    logger.debug("End \\log\\det\\Sigma_t")

    # Compute the log-likelihood of g(z_t | \mu = 0, \Sigma = I)
    # where g \sim N(0, I)
    logger.debug("Begin log-likelihood of z_t")
    sgt_loglik = innovations.loglik_std_gaussian(data=mat_z)
    logger.debug("End log-likelihood of z_t")

    # Objective function of DCC model
    loglik = sgt_loglik - 0.5 * vec_logdet_Sigma.sum()

    if normalize_loglik:
        loglik = loglik / num_sample

    if neg_loglik:
        loglik = -1 * loglik

    return loglik


def params_to_arr(
    params_dataclass: (
        innovations.ParamsZSgt | ParamsMean | ParamsUVarVol | ParamsMVarCor
    ),
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
    if isinstance(params_dataclass, innovations.ParamsZSgt):
        arr = arr.flatten()

    return arr


def _make_params_from_arr_mean(x: jpt.Array, dim: int) -> ParamsMean:
    """
    Take a vector x and split them into parameters related to the
    mean \\mu
    """
    params_mean = ParamsMean(vec_mu=x)
    return params_mean


def _make_params_from_arr_dcc_uvar_vol(x: jpt.Array, dim: int) -> ParamsUVarVol:
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


def _make_params_from_arr_dcc_mvar_cor(x: jpt.Array, dim: int) -> ParamsMVarCor:
    """
    Take a vector x and split them into \\delta parameters related to the
    DCC process.
    """
    params_mvar_cor = ParamsMVarCor(vec_delta=x)
    return params_mvar_cor


def _make_params_dcc_mvar_corQbar(
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"],
    model_dcc_distr_garch: TypeModelDcc,
    inittimecond_dcc: InitTimeConditionDcc,
) -> ParamsMVarCorQbar:
    """
    Update the \\bar{Q} parameter.
    """
    # Special treatment estimating \bar{Q}. Apply the ``variance-
    # targetting'' approach. That is, we estimate by
    # \hat{\bar{Q}} = sample moment of \hat{u}_t\hat{u}_t^{\top}.
    _, _, _, _, mat_u = calc_general_trajectories(
        mat_returns=mat_returns,
        model_dcc=model_dcc_distr_garch,
        inittimecond_dcc_distr_garch=inittimecond_dcc,
    )

    # Set \hat{\bar{Q}} = \frac{1}{T} \sum_t u_tu_t^{\top}
    _func = vmap(jnp.outer, in_axes=[0, 0])
    tens_uuT = _func(mat_u, mat_u)
    mat_Qbar = tens_uuT.mean(axis=0)

    params_mvar_corQbar = ParamsMVarCorQbar(mat_Qbar=mat_Qbar)
    return params_mvar_corQbar


def params_update(params: TypeParams, model_dcc: TypeModelDcc) -> TypeModelDcc:
    """
    Convenient function to update parameters into the main parameters dataclass.
    """
    if isinstance(params, innovations.ParamsZ):
        if isinstance(params, innovations.ParamsZGaussian) and isinstance(
            model_dcc, ModelDccGaussianGarch
        ):
            model_dcc.innov_z = params

        else:
            raise ValueError("Incorrect distribution type and model type")

    elif isinstance(params, ParamsMean):
        model_dcc.mean = params

    elif isinstance(params, ParamsUVarVol):
        model_dcc.uvar_vol = params

    elif isinstance(params, ParamsMVarCor):
        model_dcc.mvar_cor = params

    elif isinstance(params, ParamsMVarCorQbar):
        model_dcc.mvar_corQbar = params

    else:
        raise ValueError("Invalid parameter update")

    return model_dcc


def _objfun_dcc_gaussian_loglik(
    x: jpt.Array,
    make_params_from_arr: tp.Callable[[jpt.Array, int], TypeParams],
    model_dcc_gaussian_garch: ModelDccGaussianGarch,
    inittimecond_dcc_gaussian_garch: InitTimeConditionDccGaussianGarch,
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"],
) -> tp.Tuple[jpt.Float, ModelDccGaussianGarch]:
    """
    Helper function to build partial objective functions for the DCC-Gaussian-GARCH likelihood.
    """
    dim = mat_returns.shape[1]

    # Construct the dict for the parameters
    # that are to be optimized over
    _params = make_params_from_arr(x, dim)

    model_dcc_gaussian_garch = params_update(_params, model_dcc_gaussian_garch)

    logger.debug(model_dcc_gaussian_garch)
    neg_loglik = dcc_gaussian_loglik(
        mat_returns=mat_returns,
        model_dcc_gaussian_garch=model_dcc_gaussian_garch,
        inittimecond_dcc_gaussian_garch=inittimecond_dcc_gaussian_garch,
    )

    return neg_loglik, model_dcc_gaussian_garch


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


@jit
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
    return jax.lax.cond(
        l1_norm <= max_value,
        lambda _: x,
        lambda _: projection_l1_sphere(x, max_value),
        operand=None,
    )


def build_estimation_step(
    optimizer: optax.GradientTransformation,
    loss_fn: (
        tp.Callable[[jpt.Array, TypeModelDcc], tp.Tuple[jpt.Float, TypeModelDcc]]
        | tp.Callable[
            [jpt.Array, TypeModelDcc],
            tp.Tuple[jpt.Float, TypeModelDcc],
        ]
    ),
) -> tp.Callable[
    [jpt.Array | jpt.PyTree, optax.OptState, TypeModelDcc],
    tp.Tuple[jpt.Array | jpt.PyTree, optax.OptState, TypeModelDcc],
]:
    """
    Builds a function for executing a single step in the optimization
    """

    def _update(
        x: jpt.Array | jpt.PyTree, opt_state: optax.OptState, model_dcc: TypeModelDcc
    ) -> tp.Tuple[jpt.Array | jpt.PyTree, optax.OptState, TypeModelDcc]:
        grads, model_dcc = jax.grad(loss_fn, has_aux=True)(x, model_dcc)
        updates, opt_state = optimizer.update(grads, opt_state, x)
        x = optax.apply_updates(x, updates)

        return x, opt_state, model_dcc

    return _update


def fit_dcc(
    optimizer: optax.GradientTransformation,
    loss_fn: tp.Callable[
        [jpt.Array | jpt.PyTree, TypeModelDcc],
        tp.Tuple[jpt.Float, TypeModelDcc],
    ],
    x_init: jpt.Array,
    model_dcc_distr_garch: TypeModelDcc,
    numiter: int,
    lst_projection: None | tp.List[tp.Callable[[jpt.PyTree], jpt.PyTree]] = None,
) -> tp.Tuple[jpt.Bool, jpt.Float, TypeModelDcc]:
    """
    Fit DCC-distr-GARCH model
    """
    train_step = build_estimation_step(optimizer, loss_fn)

    x = x_init
    opt_state = optimizer.init(x)

    for _ in range(numiter):
        x, opt_state, model_dcc_distr_garch = train_step(
            x, opt_state, model_dcc_distr_garch
        )

        if lst_projection is not None:
            for projection in lst_projection:
                x = projection(x)

        logger.debug(f"Value function at {loss_fn(x, model_dcc_distr_garch)}")

    # Evaluate the value function
    neg_loglik_val, _ = loss_fn(x, model_dcc_distr_garch)
    valid_optimization = ~jnp.isnan(neg_loglik_val)
    return valid_optimization, neg_loglik_val, model_dcc_distr_garch


def dcc_gaussian_garch_mle(
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"],
    model_dcc_gaussian_garch: ModelDccGaussianGarch,
    inittimecond_dcc_gaussian_garch: InitTimeConditionDccGaussianGarch,
    grand_maxiter: int = 10,
    inner_maxiter: int = 10,
    optimization_schedule: tp.Callable[..., optax.Schedule] = optax.linear_schedule,
    dict_params_optimization_schedule: tp.Dict[str, float] = {
        "init_value": 1e-2,
        "end_value": 1e-4,
        "transition_steps": 10,
    },
    solver: tp.Callable[..., optax.GradientTransformation] = optax.nadam,
) -> EstimationResults:
    """
    Run DCC-Gaussian-GARCH optimization
    """
    num_sample = mat_returns.shape[0]
    dim = mat_returns.shape[1]

    logger.debug(
        f"Begin DCC-Gaussian-GARCH optimization on num_sample = {num_sample} and dim = {dim}"
    )

    #################################################################
    ## Setup partial objective functions
    #################################################################
    objfun_dcc_loglik_opt_params_mean: tp.Callable[
        [jpt.Array, ModelDccGaussianGarch], tp.Tuple[jpt.Float, ModelDccGaussianGarch]
    ] = lambda x, y: _objfun_dcc_gaussian_loglik(
        x=x,
        make_params_from_arr=_make_params_from_arr_mean,
        model_dcc_gaussian_garch=y,
        inittimecond_dcc_gaussian_garch=inittimecond_dcc_gaussian_garch,
        mat_returns=mat_returns,
    )

    objfun_dcc_loglik_opt_params_dcc_uvar_vol: tp.Callable[
        [jpt.Array, ModelDccGaussianGarch], tp.Tuple[jpt.Float, ModelDccGaussianGarch]
    ] = lambda x, y: _objfun_dcc_gaussian_loglik(
        x=x,
        make_params_from_arr=_make_params_from_arr_dcc_uvar_vol,
        model_dcc_gaussian_garch=y,
        inittimecond_dcc_gaussian_garch=inittimecond_dcc_gaussian_garch,
        mat_returns=mat_returns,
    )

    def objfun_dcc_loglik_opt_params_dcc_mvar_cor(
        x: jpt.Array, y: ModelDccGaussianGarch
    ) -> tp.Tuple[jpt.Float, ModelDccGaussianGarch]:
        return _objfun_dcc_gaussian_loglik(
            x=x,
            make_params_from_arr=_make_params_from_arr_dcc_mvar_cor,
            model_dcc_gaussian_garch=y,
            inittimecond_dcc_gaussian_garch=inittimecond_dcc_gaussian_garch,
            mat_returns=mat_returns,
        )

    # Set the projections (i.e. serve as parameter optimization constraints)
    projection_strictly_positive = lambda x: optax.projections.projection_box(
        x, lower=1e-3, upper=jnp.inf
    )
    projection_hypercube = lambda x: optax.projections.projection_box(
        x, lower=0, upper=1
    )

    #################################################################
    ## Iterative log-likelihood optimization
    #################################################################
    valid_optimization = True
    neg_loglik_val = 0.0
    iter = 0
    learning_schedule = optimization_schedule(**dict_params_optimization_schedule)
    while iter < grand_maxiter:
        learning_rate = float(learning_schedule(iter))
        optimizer = solver(learning_rate)

        ##################################################################
        ### Step 1: Optimize for the parameters of the mean \mu
        ##################################################################
        x0_mean = params_to_arr(model_dcc_gaussian_garch.mean)
        valid_optimization, neg_loglik_val, model_dcc_gaussian_garch = fit_dcc(
            optimizer=optimizer,
            loss_fn=objfun_dcc_loglik_opt_params_mean,
            x_init=x0_mean,
            model_dcc_distr_garch=model_dcc_gaussian_garch,
            numiter=inner_maxiter,
        )
        neg_loglik_val = dcc_gaussian_loglik(
            mat_returns=mat_returns,
            model_dcc_gaussian_garch=model_dcc_gaussian_garch,
            inittimecond_dcc_gaussian_garch=inittimecond_dcc_gaussian_garch,
        )
        logger.debug(
            f"..... {iter}/{grand_maxiter}: Step 1/3 --- Optimize mean \\mu params. Objective function value at {neg_loglik_val}."
        )

        ##################################################################
        ### Step 2: Optimize for the parameters of the univariate vol's
        ###         \sigma_{i,t}'s
        ##################################################################
        x0_dcc_uvar_vol = params_to_arr(model_dcc_gaussian_garch.uvar_vol)
        valid_optimization, neg_loglik_val, model_dcc_gaussian_garch = fit_dcc(
            optimizer=optimizer,
            loss_fn=objfun_dcc_loglik_opt_params_dcc_uvar_vol,
            x_init=x0_dcc_uvar_vol,
            model_dcc_distr_garch=model_dcc_gaussian_garch,
            numiter=inner_maxiter,
            lst_projection=[projection_strictly_positive],
        )
        logger.debug(
            f"..... {iter}/{grand_maxiter}: Step 2/3 --- Optimize univariate vol \\sigma params. Objective function value at {neg_loglik_val}."
        )

        #################################################################
        ## Step 3: Optimize for the parameters of the multivariate DCC
        ##         Q_t's
        #################################################################
        ## --------------------------------------------------------------
        ## Step 3(a): Optimize over the \\delta parameters in the DCC
        ## autoregressive equation
        ## --------------------------------------------------------------
        x0_dcc_mvar_cor = model_dcc_gaussian_garch.mvar_cor.vec_delta
        valid_optimization, neg_loglik_val, model_dcc_gaussian_garch = fit_dcc(
            optimizer=optimizer,
            loss_fn=objfun_dcc_loglik_opt_params_dcc_mvar_cor,
            x_init=x0_dcc_mvar_cor,
            model_dcc_distr_garch=model_dcc_gaussian_garch,
            numiter=inner_maxiter,
            lst_projection=[projection_hypercube, projection_l1_ball],
        )
        logger.debug(
            f"..... {iter}/{grand_maxiter}: Step 3a/3 --- Optimize multivariate DCC Q_t params. Objective function value at  {neg_loglik_val}."
        )

        ## --------------------------------------------------------------
        ## Step 3(b): Update \\bar{Q}
        ## --------------------------------------------------------------
        params_mvar_corQbar = _make_params_dcc_mvar_corQbar(
            mat_returns=mat_returns,
            model_dcc_distr_garch=model_dcc_gaussian_garch,
            inittimecond_dcc=inittimecond_dcc_gaussian_garch,
        )
        model_dcc_gaussian_garch.mvar_corQbar = params_mvar_corQbar
        neg_loglik_val = dcc_gaussian_loglik(
            mat_returns=mat_returns,
            model_dcc_gaussian_garch=model_dcc_gaussian_garch,
            inittimecond_dcc_gaussian_garch=inittimecond_dcc_gaussian_garch,
        )

        logger.debug(
            f"..... {iter}/{grand_maxiter}: Step 3b/3 --- Optimize multivariate DCC Q_t params. Objective function value at  {neg_loglik_val}."
        )

        #################################################################
        ## Update iteration
        #################################################################
        iter += 1

        if valid_optimization:
            continue
        else:
            logger.warn(
                f"Reached invalid parameter update at iter = {iter} with learning rate = {learning_rate}"
            )
            estimation_res = EstimationResults(
                valid_optimization=valid_optimization,
                neg_loglik_val=neg_loglik_val,
                dcc_model=model_dcc_gaussian_garch,
            )
            return estimation_res

    logger.debug("DCC optimization complete")
    estimation_res = EstimationResults(
        valid_optimization=valid_optimization,
        neg_loglik_val=neg_loglik_val,
        dcc_model=model_dcc_gaussian_garch,
    )
    return estimation_res


def gen_simulation_dcc_gaussian_garch(
    num_sample: int, dim: int, seed: int | None = None
) -> SimulatedReturns:
    """
    Generate a simulation of the DCC-Gaussian-GARCH model and save the
    results.
    """
    #################################################################
    ## Setup
    #################################################################
    # Random initial seed based on current time
    if seed is None:
        seed = utils.gen_seed_number()
    hashid = uuid.uuid4().hex
    str_id = utils.gen_str_id(num_sample=num_sample, dim=dim, hashid=hashid)

    key = random.key(seed)
    rng = np.random.default_rng(seed)

    #################################################################
    ## Parameters for standard Gaussian
    #################################################################
    mean = jnp.repeat(0, dim)
    cov = jnp.eye(dim)
    params_innov_z_true = innovations.ParamsZGaussian(mean=mean, cov=cov)

    #################################################################
    ## Parameters for DCC-GARCH
    #################################################################
    # Parameters for the mean returns vector
    params_mean_true = ParamsMean(vec_mu=jnp.array(rng.uniform(0, 1, dim) / 50))

    # Params for DCC -- univariate vols
    params_uvar_vol_true = ParamsUVarVol(
        vec_omega=jnp.array(rng.uniform(0, 1, dim)),
        vec_beta=jnp.array(rng.uniform(0, 1, dim)),
        vec_alpha=jnp.array(rng.uniform(0, 1, dim)),
        vec_psi=jnp.array(rng.uniform(0, 1, dim)),
    )
    # Params for DCC -- multivariate Q
    params_mvar_cor_true = ParamsMVarCor(
        vec_delta=jnp.array([0.007, 0.930]),
    )
    params_mvar_corQbar_true = ParamsMVarCorQbar(
        mat_Qbar=generate_random_cov_mat(key=key, dim=dim),
    )

    # Initial t = 0 conditions for the DCC Q_t process
    subkeys = random.split(key, 2)
    mat_Sigma_init_t0 = generate_random_cov_mat(key=subkeys[0], dim=dim)
    mat_Q_init_t0 = generate_random_cov_mat(key=subkeys[1], dim=dim)
    inittimecond_dcc_true = InitTimeConditionDcc(
        mat_Sigma_init_t0=mat_Sigma_init_t0, mat_Q_init_t0=mat_Q_init_t0
    )

    # Put everything together
    model_dcc_gaussian_garch_true = ModelDccGaussianGarch(
        mean=params_mean_true,
        uvar_vol=params_uvar_vol_true,
        mvar_cor=params_mvar_cor_true,
        mvar_corQbar=params_mvar_corQbar_true,
        innov_z=params_innov_z_true,
    )
    inittimecond_dcc_gaussian_garch = InitTimeConditionDccGaussianGarch(
        dcc=inittimecond_dcc_true,
        mat_Q_init_t0=mat_Q_init_t0,
        mat_Sigma_init_t0=mat_Sigma_init_t0,
    )

    #################################################################
    ## Simulate DCC-Gaussian-GARCH
    #################################################################
    simreturns = simulate_dcc_gaussian_garch(
        seed=seed,
        hashid=hashid,
        key=key,
        dim=dim,
        num_sample=num_sample,
        model_dcc_gaussian_garch=model_dcc_gaussian_garch_true,
        inittimecond_dcc_gaussian_garch=inittimecond_dcc_gaussian_garch,
    )
    return simreturns


def calc_estimation_dcc_gaussian_garch(
    mat_returns: jpt.Float[jpt.Array, "num_sample dim"], seed: int | None = None
) -> EstimationResults:
    dim = mat_returns.shape[1]

    if seed is None:
        seed = utils.gen_seed_number()
    key = random.key(seed)
    rng = np.random.default_rng(seed)

    #################################################################
    ## Initial parameter guesses
    #################################################################
    # Innovations parameters
    mean = jnp.repeat(0, dim)
    cov = jnp.eye(dim)
    params_innov_z_known = innovations.ParamsZGaussian(mean=mean, cov=cov)

    # Initial guess for parameters for the mean returns vector
    # params_mean_init_guess = dcc.ParamsMean(vec_mu=jnp.array(rng.uniform(0, 1, DIM) / 50))
    params_mean_init_guess = ParamsMean(vec_mu=jnp.array(rng.uniform(0, 1, dim) / 50 ))

    # Initial guess for params for DCC -- univariate vols
    params_uvar_vol_init_guess = ParamsUVarVol(
        vec_omega=jnp.array(rng.uniform(0, 1, dim)),
        vec_beta=jnp.array(rng.uniform(0, 1, dim)),
        vec_alpha=jnp.array(rng.uniform(0, 1, dim)),
        vec_psi=jnp.array(rng.uniform(0, 1, dim)),
    )

    # Initial guess for params for DCC -- multivariate Q
    # FIX: Need to randomize this
    params_mvar_cor_init_guess = ParamsMVarCor(
        vec_delta=jnp.array([0.154, 0.530]),
    )
    params_mvar_corQbar_init_guess = ParamsMVarCorQbar(
        mat_Qbar=generate_random_cov_mat(key=key, dim=dim),
    )

    # Package all the initial guess DCC params together
    guess_model_dcc_gaussian_garch = ModelDccGaussianGarch(
        mean=params_mean_init_guess,
        uvar_vol=params_uvar_vol_init_guess,
        mvar_cor=params_mvar_cor_init_guess,
        mvar_corQbar=params_mvar_corQbar_init_guess,
        innov_z=params_innov_z_known,
    )

    # Initial t = 0 conditions for the DCC Q_t process
    subkeys = random.split(key, 2)
    mat_Sigma_init_t0_guess = generate_random_cov_mat(key=subkeys[0], dim=dim)
    mat_Q_init_t0_guess = generate_random_cov_mat(key=subkeys[1], dim=dim)
    inittimecond_dcc_guess = InitTimeConditionDcc(
        mat_Sigma_init_t0=mat_Sigma_init_t0_guess, mat_Q_init_t0=mat_Q_init_t0_guess
    )

    inittimecond_dcc_gaussian_garch_guess = InitTimeConditionDccGaussianGarch(
        dcc=inittimecond_dcc_guess,
        mat_Q_init_t0=mat_Q_init_t0_guess,
        mat_Sigma_init_t0=mat_Sigma_init_t0_guess,
    )

    #################################################################
    ## Maximum likelihood estimation
    #################################################################
    estimation_res = dcc_gaussian_garch_mle(
        mat_returns=mat_returns,
        model_dcc_gaussian_garch=guess_model_dcc_gaussian_garch,
        inittimecond_dcc_gaussian_garch=inittimecond_dcc_gaussian_garch_guess,
    )
    return estimation_res


if __name__ == "__main__":
    pass
