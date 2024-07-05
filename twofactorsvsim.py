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

import numpy as np

import typing as tp
import time
import uuid
import pathlib
import logging
from datetime import datetime
import pickle
import argparse
import os

from matplotlib import pyplot as plt


import dataclasses
from dataclasses import dataclass

import utils

jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod


@dataclass
class ParamsTwoFactorSV:
    mu: jpt.Float = 0.030
    beta: jpt.Array = jnp.array([-1.200, 0.040, 1.500])
    alpha: jpt.Array = jnp.array([-0.00137, -1.386])
    phi: jpt.Float = 0.250
    rho: jpt.Array = jnp.array([-0.300, -0.300])

    # Parameters for the diurnal U-shape \\sigma_{ut} function
    A: jpt.Float = 0.75
    B: jpt.Float = 0.25
    C: jpt.Float = 0.88929198
    a: jpt.Float = 10.0
    b: jpt.Float = 10.0


# HACK:
seed = 1234567
if seed is None:
    seed = utils.gen_seed_number()
key = random.key(seed)
rng = np.random.default_rng(seed)

TimeSequence: tp.TypeAlias = jpt.Float[jpt.Array, "num_discr"]
TimeStepSize: tp.TypeAlias = jpt.Float
Dimension: tp.TypeAlias = jpt.Integer
NumDiscretization: tp.TypeAlias = jpt.Integer


def calc_dW(
    key: KeyArrayLike, delta_t: TimeStepSize, num_discr: NumDiscretization, d: Dimension
) -> jpt.Float[jpt.Array, "num_discr d"]:
    """
    Sample [dW_{1,t}, \\ldots, dW{d,t}]
    """
    mean = jnp.repeat(0, d)
    cov = delta_t * jnp.eye(d)
    shape = (num_discr,)

    return jax.random.multivariate_normal(key=key, mean=mean, cov=cov, shape=shape)


def calc_dnu_1_squared(
    ts: TimeSequence,
    delta_t: TimeStepSize,
    alpha_1: jpt.Float,
    dW_1: jpt.Float[jpt.Array, "ts.size"],
    nu_1_init: jpt.Float,
) -> jpt.Float[jpt.Array, "ts.size"]:
    """
    Simulate d\\nu_{1t}^2 = \\alpha_1 \\nu_{1t}^2 dt + dW_{1t}
    """

    def _drift(nu_1_squared_t_minus_1):
        return alpha_1 * nu_1_squared_t_minus_1

    def _diffusion(nu_1_squared_t_minus_1):
        return 1.0

    # def _calc_nu_1_squared_t(dW_1t, nu_1_squared_t_minus_1):
    #     nu_1_squared_t = (
    #         nu_1_squared_t_minus_1
    #         + _drift(nu_1_squared_t_minus_1) * delta_t
    #         + _diffusion(nu_1_squared_t_minus_1) * dW_1t
    #     )
    #     return nu_1_squared_t, nu_1_squared_t

    def _calc_nu_1_squared_t(carry, xs):
        dW_1t = xs
        nu_1_squared_t_minus_1 = carry

        nu_1_squared_t = (
            nu_1_squared_t_minus_1
            + _drift(nu_1_squared_t_minus_1=nu_1_squared_t_minus_1) * delta_t
            + _diffusion(nu_1_squared_t_minus_1=nu_1_squared_t_minus_1) * dW_1t
        )
        carry = nu_1_squared_t
        return carry, carry


    # vec_nu_1_squared = jnp.zeros(ts.size)
    # vec_nu_1_squared = vec_nu_1_squared.at[0].set(nu_1_init**2)
    #
    # for ii in range(1, ts.size):
    #     nu_1_squared_t_minus_1 = vec_nu_1_squared[ii - 1]
    #
    #     nu_1_squared_t = (
    #         nu_1_squared_t_minus_1
    #         + _drift(nu_1_squared_t_minus_1) * delta_t
    #         + _diffusion(nu_1_squared_t_minus_1) * dW_1[ii]
    #     )
    #     vec_nu_1_squared = vec_nu_1_squared.at[ii].set(nu_1_squared_t)


    nu_1_squared_init = nu_1_init**2
    _, vec_nu_1_squared = jax.lax.scan(_calc_nu_1_squared_t, init = nu_1_squared_init, xs = dW_1[1:])
    vec_nu_1_squared = jnp.insert(vec_nu_1_squared, obj=0, values=nu_1_squared_init, axis = 0)

    return vec_nu_1_squared


# @jit
def calc_dnu_2_squared(
    ts: TimeSequence,
    delta_t: jpt.Float,
    alpha_2: jpt.Float,
    phi: jpt.Float,
    dW_2: jpt.Float[jpt.Array, "ts.size"],
    nu_2_init: jpt.Float,
) -> jpt.Float[jpt.Array, "ts.size"]:
    """
    Simulate d\\nu_{2t}^2 = \\alpha_2 \\nu_{2t}^2 dt + (1 + \\phi \\nu_{2t}^2)dW_{1t}
    """


    def _drift(nu_2_squared_t_minus_1):
        return alpha_2 * nu_2_squared_t_minus_1

    def _diffusion(nu_2_squared_t_minus_1):
        return 1 + phi * nu_2_squared_t_minus_1

    # def _calc_nu_2_squared_t(dW_2t, nu_2_squared_t_minus_1):
    #     nu_2_squared_t = (
    #         nu_2_squared_t_minus_1
    #         + _drift(nu_2_squared_t_minus_1) * delta_t
    #         + _diffusion(nu_2_squared_t_minus_1) * dW_2t
    #     )
    #     return nu_2_squared_t, nu_2_squared_t

    def _calc_nu_2_squared_t(carry, xs):
        dW_2t = xs
        nu_2_squared_t_minus_1 = carry

        nu_2_squared_t = (
            nu_2_squared_t_minus_1
            + _drift(nu_2_squared_t_minus_1=nu_2_squared_t_minus_1) * delta_t
            + _diffusion(nu_2_squared_t_minus_1=nu_2_squared_t_minus_1) * dW_2t
        )
        carry = nu_2_squared_t
        return carry, carry


    # vec_nu_2_squared = jnp.zeros(ts.size)
    # vec_nu_2_squared = vec_nu_2_squared.at[0].set(nu_2_init**2)
    #
    # for ii in range(1, ts.size):
    #     nu_2_squared_t_minus_1 = vec_nu_2_squared[ii - 1]
    #
    #     nu_2_squared_t = (
    #         nu_2_squared_t_minus_1
    #         + _drift(nu_2_squared_t_minus_1) * delta_t
    #         + _diffusion(nu_2_squared_t_minus_1) * dW_2[ii]
    #     )
    #     vec_nu_2_squared = vec_nu_2_squared.at[ii].set(nu_2_squared_t)
    #
    nu_2_squared_init = nu_2_init**2

    _, vec_nu_2_squared = jax.lax.scan(_calc_nu_2_squared_t, init = nu_2_squared_init, xs = dW_2[1:])
    vec_nu_2_squared = jnp.insert(arr=vec_nu_2_squared, obj=0, values=nu_2_squared_init, axis=0)

    return vec_nu_2_squared


# @jit
def spline_exponential(x: jpt.Float) -> jpt.Float:
    """
    Define the s-exp function of Chernov, Gallant, Ghysels and Tauchen (2013)
    "Alternative models for stock price dynamics" as:
    sexp(u) = exp(u) if u \\le u_0 = \\log(1.5), =
    (exp(u0) / \\sqrt{u0}) \\sqrt{u0 - u0^2 + u^2}
    """
    x0 = jnp.log(1.5)

    return jax.lax.select(
        x <= x0,
        jnp.exp(x),
        (jnp.exp(x0) / jnp.sqrt(x0)) * (jnp.sqrt(x0 - x0**2 + x**2)),
    )


# @jit
def calc_nu_squared(
    beta,
    vec_nu_1_squared: jpt.Float[jpt.Array, "ts.size"],
    vec_nu_2_squared: jpt.Float[jpt.Array, "ts.size"],
) -> jpt.Float[jpt.Array, "ts.size"]:
    """
    Calculate \\nu_t^2 = s-exp( \\beta_0 + \\beta_1\\nu_{1t}^2 + \\beta_2\\nu_{2t}^2 )
    """
    val = beta[0] + beta[1] * vec_nu_1_squared + beta[2] * vec_nu_2_squared
    return spline_exponential(val)


# @jit
def calc_sigma_u(
    tt: jpt.Float, A: jpt.Float, B: jpt.Float, C: jpt.Float, a: jpt.Float, b: jpt.Float
) -> jpt.Float:
    """
    Diurnal U-shape function \\sigma_{ut} = C + A e^{-at} + B e^{-b(1 - t)}
    """
    return C + A * jnp.exp(-a * tt) + B * jnp.exp(-b * (1 - tt))


def calc_dlogS(
    ts: jpt.Float[jpt.Array, "ts.size"],
    delta_t: jpt.Float,
    params: ParamsTwoFactorSV,
    dW : jpt.Float[jpt.Array, "ts.size"],
    price_init: jpt.Float,
    nu_init: jpt.Float,
) -> jpt.Float[jpt.Array, "ts.size"]:

    def _drift():
        return params.mu

    def _diffusion_1(i):
        """
        Diffusion term \\rho_1\\sigma_{ut}\\nu_t
        """
        return params.rho[0] * vec_sigma_u[i] * vec_nu[i]

    def _diffusion_2(i):
        """
        Diffusion term \\rho_2\\sigma_{ut}\\nu_t
        """
        return params.rho[1] * vec_sigma_u[i] * vec_nu[i]

    def _diffusion_3(i):
        """
        Diffusion term \\sqrt{1 - \\rho_1^2 - \\rho_2^2}\\sigma_{ut}\\nu_t
        """
        return (
            jnp.sqrt(1 - params.rho[1] ** 2 - params.rho[2] ** 2)
            * vec_sigma_u[i]
            * vec_nu[i]
        )

    def _calc_logS_t(ii, logS_t_minus_1):
        """
        Convenient solution for transitioning to \\log S_t
        """
        logS_t = (
            logS_t_minus_1
            + _drift() * delta_t
            + _diffusion_1(ii) * dW[ii, 0]
            + _diffusion_2(ii) * dW[ii, 1]
            + _diffusion_3(ii) * dW[ii, 2]
        )
        # vec_logS = vec_logS.at[ii].set(logS_t)

        return ii + 1, logS_t

    def _diffusion_11(sigma_u_t, nu_t):
        """
        Diffusion term \\rho_1\\sigma_{ut}\\nu_t
        """
        return params.rho[0] * sigma_u_t * nu_t

    def _diffusion_22(sigma_u_t, nu_t):
        """
        Diffusion term \\rho_2\\sigma_{ut}\\nu_t
        """
        return params.rho[1] * sigma_u_t * nu_t

    def _diffusion_33(sigma_u_t, nu_t):
        """
        Diffusion term \\sqrt{1 - \\rho_1^2 - \\rho_2^2}\\sigma_{ut}\\nu_t
        """
        return (
            jnp.sqrt(1 - params.rho[1] ** 2 - params.rho[2] ** 2)
            * sigma_u_t
            * nu_t
        )

    def _calc_logS_tt(carry, xs):
        dW_t, sigma_u_t_minus_1, nu_t_minus_1 = xs
        logS_t_minus_1 = carry

        logS_t = (
            logS_t_minus_1
            + _drift() * delta_t
            + _diffusion_11(sigma_u_t_minus_1, nu_t_minus_1) * dW_t[0]
            + _diffusion_22(sigma_u_t_minus_1, nu_t_minus_1) * dW_t[1]
            + _diffusion_33(sigma_u_t_minus_1, nu_t_minus_1) * dW_t[2]
        )
        carry = logS_t
        return carry, carry

    # Compute \\nu_t's
    vec_nu_1_squared = calc_dnu_1_squared(
        ts=ts,
        delta_t=delta_t,
        alpha_1=params.alpha[0],
        dW_1=dW[:, 0],
        nu_1_init=nu_init[0],
    )
    vec_nu_2_squared = calc_dnu_2_squared(
        ts=ts,
        delta_t=delta_t,
        alpha_2=params.alpha[1],
        phi=params.phi,
        dW_2=dW[:, 1],
        nu_2_init=nu_init[1],
    )
    vec_nu_squared = calc_nu_squared(
        beta=params.beta,
        vec_nu_1_squared=vec_nu_1_squared,
        vec_nu_2_squared=vec_nu_2_squared,
    )
    vec_nu = jnp.sqrt(vec_nu_squared)

    # Compute \\sigma_{ut}'s
    vec_sigma_u = calc_sigma_u(
        ts, A=params.A, B=params.B, C=params.C, a=params.a, b=params.b
    )



    # # Compute \\logS_t's
    # vec_logS = jnp.zeros(ts.size)
    # vec_logS = vec_logS.at[0].set(jnp.log(price_init))
    #
    # for ii in range(1, ts.size):
    #     logS_t_minus_1 = vec_logS[ii - 1]
    #
    #     logS_t = (
    #         logS_t_minus_1
    #         + _drift() * delta_t
    #         + _diffusion_1(ii - 1) * dW[ii - 1, 0]
    #         + _diffusion_2(ii - 1) * dW[ii - 1, 1]
    #         + _diffusion_3(ii - 1) * dW[ii - 1, 2]
    #     )
    #     vec_logS = vec_logS.at[ii].set(logS_t)



    logS_init =jnp.log(price_init)
    xs = (dW, vec_sigma_u, vec_nu)
    _, vec_logS = jax.lax.scan(_calc_logS_tt, init = logS_init, xs = xs)
    vec_logS = jnp.insert(vec_logS, 0, logS_init, axis = 0)
    vec_logS = jnp.delete(vec_logS, -1, axis = 0)

    return vec_logS


d = 3
t_init = 0
t_end = 1
burn_in_discr = 0
num_discr = 23400
tot_discr = burn_in_discr + num_discr
delta_t = float((t_end - t_init)) / tot_discr
ts = jnp.arange(t_init, t_end + delta_t, delta_t)
assert ts.size == tot_discr + 1

params = ParamsTwoFactorSV()

dW = calc_dW(key=key, delta_t=delta_t, num_discr=ts.size, d=d)

# Initial conditions
nu_1_init = jax.random.normal(key=key) * jnp.sqrt(-1 / (2 * params.alpha[0]))
nu_2_init = 0.0
nu_init = jnp.array([nu_1_init, nu_2_init])
price_init = 1.0


vec_logS = calc_dlogS(
    ts=ts, delta_t=delta_t, params=params, dW=dW, price_init=price_init, nu_init=nu_init
)
vec_S = jnp.exp(vec_logS)

plt.plot(vec_S[burn_in_discr:])
plt.show()
