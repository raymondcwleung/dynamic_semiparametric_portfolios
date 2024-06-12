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

import logging

import typing as tp


import itertools

# import optax
import jaxopt

import numpy as np
from numpy._typing import ArrayLike, NDArray
import scipy
import matplotlib.pyplot as plt

import sgt

# HACK:
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod
jax.config.update("jax_debug_nans", True)  # Should disable in full prod


logger = logging.getLogger(__name__)


def _generate_random_cov_mat(key: KeyArrayLike, dim: int) -> Array:
    """
    Generate a random covariance-variance matrix (i.e. symmetric and PSD).
    """
    mat_A = jax.random.normal(key, shape=(dim, dim)) * 0.25
    mat_sigma = jnp.transpose(mat_A) @ mat_A

    return mat_sigma


def _calc_unexpected_excess_rtn(mat_Sigma: Array, vec_z: Array) -> Array:
    """
    Return \\epsilon = \\Sigma^{1/2} z.

    Given a covariance matrix \\Sigma and an innovation
    vector z, return the unexpected excess returns vector
    \\epsilon = \\Sigma^{1/2} z, where \\Sigma^{1/2} is
    taken to be the Cholesky decomposition of \\Sigma.
    """
    mat_Sigma_sqrt = jscipy.linalg.cholesky(mat_Sigma)
    vec_epsilon = mat_Sigma_sqrt @ vec_z

    return vec_epsilon


def _calc_normalized_unexpected_excess_rtn(
    vec_sigma: Array, vec_epsilon: Array
) -> Array:
    """
    Return u = D^{-1} \\epsilon
    """
    mat_inv_D = jnp.diag(1 / vec_sigma)

    return mat_inv_D @ vec_epsilon


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


def _calc_mat_Gamma(mat_Q: Array) -> Array:
    """
    Return the \\Gamma_t matrix of the DCC model
    """
    mat_Qstar_inv_sqrt = jnp.diag(jnp.diag(mat_Q) ** (-1 / 2))
    mat_Gamma = mat_Qstar_inv_sqrt @ mat_Q @ mat_Qstar_inv_sqrt

    return mat_Gamma


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
        vec_sigma2_t = (
            vec_omega
            + vec_beta * vec_sigma_t_minus_1**2
            + vec_alpha * vec_epsilon_t_minus_1**2
            + vec_psi * vec_epsilon_t_minus_1**2 * (vec_epsilon_t_minus_1 < 0)
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


def simulate_returns(
    seed: int,
    dim: int,
    num_sample: int,
    dict_params_mean,
    dict_params_z,
    dict_params_dcc,
):
    key = random.key(seed)
    rng = np.random.default_rng(seed)

    # Simulate the innovations
    data_z = sgt.sample_mvar_sgt(key=key, num_sample=num_sample, **dict_params_z)

    # Simulate a DCC model and obtain \epsilon_t and \Sigma_t
    mat_epsilon, tns_Sigma = simulate_dcc(key=key, data_z=data_z, **dict_params_dcc)

    # Set the asset mean
    vec_mu = dict_params_mean["vec_mu"]

    # Asset returns
    mat_returns = vec_mu + mat_epsilon

    # Sanity checks
    if mat_returns.shape != (num_sample, dim):
        logger.error("Incorrect shape for the simulated returns.")

    return mat_returns


def _calc_innovations(vec_returns: Array, mat_Sigma: Array) -> Array:
    """
    Return z_t = \\Sigma_t^{-1/2} R_t, where we are given
    returns {R_t} and conditional covariances {\\Sigma_t},
    """
    mat_Sigma_sqrt = jnp.linalg.cholesky(mat_Sigma)
    vec_z = jnp.linalg.solve(mat_Sigma_sqrt, vec_returns)

    return vec_z


def calc_innovations(mat_returns: Array, tns_Sigma: Array) -> Array:
    """
    Calculate innovations over the full sample.
    """
    _func = vmap(_calc_innovations, in_axes=[0, 0])
    mat_z = _func(mat_returns, tns_Sigma)

    return mat_z


if __name__ == "__main__":
    seed = 1234567
    key = random.key(seed)
    rng = np.random.default_rng(seed)
    num_sample = int(1e3)
    dim = 3
    num_cores = 8

    # Parameters for the mean returns vector
    dict_params_mean = {"vec_mu": rng.uniform(0, 1, dim) / 50}

    # Params for z \sim SGT
    dict_params_z = {
        "vec_lbda": rng.uniform(-0.25, 0.25, dim),
        "vec_p0": rng.uniform(2, 4, dim),
        "vec_q0": rng.uniform(2, 4, dim),
    }

    # Params for DCC
    dict_params_dcc = {
        "vec_omega": rng.uniform(0, 1, dim) / 2,
        "vec_beta": rng.uniform(0, 1, dim) / 3,
        "vec_alpha": rng.uniform(0, 1, dim) / 10,
        "vec_psi": rng.uniform(0, 1, dim) / 5,
        # vec_delta_true = rng.uniform(0, 1, 2)
        # Ensure \delta_1, \delta_2 \in [0,1] and \delta_1 + \delta_2 \le 1
        "vec_delta": np.array([0.007, 0.930]),
        "mat_Qbar": _generate_random_cov_mat(key=key, dim=dim) / 10,
    }

    mat_returns = simulate_returns(
        seed=seed,
        dim=dim,
        num_sample=num_sample,
        dict_params_mean=dict_params_mean,
        dict_params_z=dict_params_z,
        dict_params_dcc=dict_params_dcc,
    )
