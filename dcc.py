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


seed = 1234567
key = random.key(seed)
rng = np.random.default_rng(seed)
num_sample = int(1e3)
dim = 3
num_cores = 8

# Params for z \sim SGT
vec_lbda_true = rng.uniform(-0.25, 0.25, dim)
vec_p0_true = rng.uniform(2, 4, dim)
vec_q0_true = rng.uniform(2, 4, dim)

# Params for DCC
vec_omega_true = rng.uniform(0, 1, dim) / 2
vec_beta_true = rng.uniform(0, 1, dim) / 3
vec_alpha_true = rng.uniform(0, 1, dim) / 10
vec_psi_true = rng.uniform(0, 1, dim) / 5
# vec_delta_true = rng.uniform(0, 1, 2)
# Ensure \delta_1, \delta_2 \in [0,1] and \delta_1 + \delta_2 \le 1
vec_delta_true = np.array([0.007, 0.930])
mat_Qbar_true = _generate_random_cov_mat(key=key, dim=dim) / 10

data_z = sgt.sample_mvar_sgt(
    key=key,
    num_sample=num_sample,
    vec_lbda=vec_lbda_true,
    vec_p0=vec_p0_true,
    vec_q0=vec_q0_true,
    num_cores=num_cores,
)


vec_omega = vec_omega_true
vec_beta = vec_beta_true
vec_alpha = vec_alpha_true
vec_psi = vec_psi_true
vec_delta = vec_delta_true
mat_Qbar = mat_Qbar_true

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


asset_i = 2
yy = [lst_sigma[tt][asset_i] for tt in range(len(lst_sigma))]
yy = np.array(yy)
xx = range(len(lst_sigma))
plt.plot(xx, yy)
plt.show()
