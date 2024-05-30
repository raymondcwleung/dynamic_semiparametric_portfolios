import jax.numpy as jnp
import jax.scipy as jscipy
import jax.scipy.optimize
from jax import grad, jit, vmap
from jax import random
import jax
import jax.test_util


# import optax
import jaxopt

import numpy as np
import scipy
import matplotlib.pyplot as plt


import time

# HACK:
jax.config.update("jax_default_device", jax.devices("cpu")[0])
# jax.config.update("jax_enable_x64", True)


def _density_normalized_multivariate_skew_normal(x, mat_omega_bar, vec_alpha):
    """
    Density of a SN_d(0, \bar{\Omega}, \alpha) random vector.
    See Azzalini (2014, Chapter 5), eq (5.1).
    """
    dim = jnp.size(x)

    # \phi_d(x ; \Omega)
    psi_d = jscipy.stats.multivariate_normal.pdf(
        x=x, mean=jnp.repeat(0, dim), cov=mat_omega_bar
    )
    # \Phi(\alpha^\top x)
    phi = jscipy.stats.norm.cdf(x=jnp.inner(vec_alpha, x))

    return 2 * psi_d * phi


def _sample_normalized_multivariate_skew_normal(key, mat_omega_bar, vec_alpha):
    """
    Sample a SN_d(0, \bar{\Omega}, \alpha) random vector by using its
    stochastic representation. See Azzalini (2014, Section 5.1.3).
    """
    dim = jnp.size(vec_alpha)

    # Draw X_0 \sim N_d(0, \bar{\Omega})
    vec_x0 = jax.random.multivariate_normal(
        key=key, mean=jnp.repeat(0, dim), cov=mat_omega_bar
    )

    # Draw T \sim N(0,1) that is independent of X_0
    key, subkey = random.split(key)
    t = jax.random.normal(key)

    # Set
    # Z = X_0,  if T > \alpha^\top X_0 \\
    #   = -X_0, if otherwise.
    if t > jnp.inner(vec_alpha, vec_x0):
        vec_z = vec_x0
    else:
        vec_z = -vec_x0

    return vec_z


def _density_multivariate_student_t(x, mat_sigma, nu):
    """
    Density of a multivariate Student t's distribution t_d(x ; \Sigma, \nu).
    See Azzalini (2014) eq (6.9).
    """
    pi = jnp.pi
    gamma = jax.scipy.special.gamma
    det = jnp.linalg.det
    inner = jnp.inner
    solve = jnp.linalg.solve

    dim = jnp.size(x)

    _first_term = gamma((nu + dim) / 2) / (
        (nu * pi) ** (dim / 2) * gamma(nu / 2) * det(mat_sigma) ** (1 / 2)
    )

    # Compute x^\top \Sigma^{-1} x
    x_sigmainv_x = inner(x, solve(mat_sigma, x))

    _second_term = (1 + x_sigmainv_x / nu) ** (-(nu + dim) / 2)

    return _first_term * _second_term


def _density_normalized_multivariate_skew_t(x, mat_omega_bar, vec_alpha, nu):
    """
    Density of a ST_d(0, \bar{\Omega}, \alpha, \nu) random vector.
    See Azzalini (2014, Section 6.2.1)
    """
    dim = jnp.size(x)

    # t_d(z; \bar\Omega}, nu)


dim = 2
key = random.key(0)
vec_alpha = 2 * jax.random.uniform(key, shape=(dim,)) - 1
mat_omega_bar = jnp.identity(dim)

lst_blah = []
for i in range(10):
    key, subkey = random.split(key)
    blah = _sample_normalized_multivariate_skew_normal(key, mat_omega_bar, vec_alpha)

    lst_blah.append(blah)
