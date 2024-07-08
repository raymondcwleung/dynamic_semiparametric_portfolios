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

import time

from matplotlib import pyplot as plt


import dataclasses
from dataclasses import dataclass

import utils

# jax.config.update("jax_enable_x64", True)  # Should use x32 in full prod
# jax.config.update("jax_debug_nans", True)  # Should disable in full prod


TimeSequence: tp.TypeAlias = jpt.Float[jpt.Array, "num_discr"]
TimeStepSize: tp.TypeAlias = jpt.Float
Dimension: tp.TypeAlias = jpt.Integer
NumDiscretization: tp.TypeAlias = jpt.Integer
NumReplications: tp.TypeAlias = jpt.Integer
DeltaT: tp.TypeAlias = jpt.Float
SimulationPathOneDay: tp.TypeAlias = jpt.Float[jpt.Array, "num_discr"]
SimulationPathOneDay3: tp.TypeAlias = jpt.Float[jpt.Array, "num_discr"]
SimulationPathAllDays3: tp.TypeAlias = jpt.Float[
    jpt.Array, "num_discr*total_num_days 3"
]
SimulationPathAllDaysAllEpochs: tp.TypeAlias = jpt.Float[
    jpt.Array, "total_num_epochs num_discr*total_num_days"
]
SimulationPathAllDaysAllEpochs3: tp.TypeAlias = jpt.Float[
    jpt.Array, "total_num_epochs num_discr*total_num_days 3"
]


@dataclass
class ParamsTwoFactorSV:
    seed: int = 12345
    key: KeyArrayLike = dataclasses.field(init=False)

    # Total number of simulation epochs to run
    total_num_epochs: int = 5

    # Total number of days to simulate
    total_num_days: int = 1000

    # Initial values
    init_price: float = 1.0  # Initial asset price
    init_nu: jpt.Array = dataclasses.field(init=False)

    # These parameters should essentially be fixed
    t_init: float = 0
    t_end: float = 1
    d: int = 3  # Dimension is fixed as per two-factor SV model for single asset
    num_time_units_per_day: int = 23400  # i.e. Number of seconds in one trading day

    # Time steps
    delta_t: TimeStepSize = dataclasses.field(init=False)
    ts: TimeSequence = dataclasses.field(init=False)

    #  Parameters for the d\\logS, d\\nu_1 and d\\nu_2 processes
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

    def __post_init__(self):
        # Set JAX key
        self.key = random.key(self.seed)

        # Set random initial conditions for \\nu_1, \\nu_2
        init_nu_1 = jax.random.normal(key=self.key) * jnp.sqrt(-1 / (2 * self.alpha[0]))
        init_nu_2 = 0.0
        self.init_nu = jnp.array([init_nu_1, init_nu_2])

        # Time steps
        self.delta_t = float((self.t_end - self.t_init)) / self.num_time_units_per_day
        self.ts = jnp.linspace(
            start=self.t_init,
            stop=self.t_end,
            num=self.num_time_units_per_day,
            endpoint=True,
        )


def calc_dW(
    key: KeyArrayLike,
    delta_t: TimeStepSize,
    num_discr: NumDiscretization,
    num_replications: NumReplications,
    d: Dimension,
) -> jpt.Float[jpt.Array, "num_discr num_replications d"]:
    """
    Sample [dW_{1,t}, \\ldots, dW{d,t}]
    """
    return jnp.sqrt(delta_t) * jax.random.normal(
        key=key, shape=(num_replications, num_discr, d)
    )


@jit
def calc_dnu_1(
    delta_t: TimeStepSize,
    alpha_1: jpt.Float,
    dW_1: SimulationPathOneDay,
    nu_1_init: jpt.Float,
) -> SimulationPathOneDay:
    """
    Simulate d\\nu_{1t} = \\alpha_1 \\nu_{1t} dt + dW_{1t}
    """

    def _drift(nu_1_t_minus_1):
        return alpha_1 * nu_1_t_minus_1

    def _diffusion():
        return 1.0

    def _calc_nu_1_t(carry, xs):
        dW_1t = xs
        nu_1_t_minus_1 = carry

        nu_1_t = (
            nu_1_t_minus_1
            + _drift(nu_1_t_minus_1=nu_1_t_minus_1) * delta_t
            + _diffusion() * dW_1t
        )
        carry = nu_1_t
        return carry, carry

    _, vec_nu_1 = jax.lax.scan(_calc_nu_1_t, init=nu_1_init, xs=dW_1[1:])
    vec_nu_1 = jnp.insert(vec_nu_1, obj=0, values=nu_1_init, axis=0)

    return vec_nu_1


@jit
def calc_dnu_2(
    delta_t: jpt.Float,
    alpha_2: jpt.Float,
    phi: jpt.Float,
    dW_2: SimulationPathOneDay,
    nu_2_init: jpt.Float,
) -> SimulationPathOneDay:
    """
    Simulate d\\nu_{2t} = \\alpha_2 \\nu_{2t} dt + (1 + \\phi \\nu_{2t})dW_{2t}
    """

    def _drift(nu_2_t_minus_1):
        return alpha_2 * nu_2_t_minus_1

    def _diffusion(nu_2_t_minus_1):
        return 1 + phi * nu_2_t_minus_1

    def _calc_nu_2_t(carry, xs):
        dW_2t = xs
        nu_2_t_minus_1 = carry

        nu_2_t = (
            nu_2_t_minus_1
            + _drift(nu_2_t_minus_1=nu_2_t_minus_1) * delta_t
            + _diffusion(nu_2_t_minus_1=nu_2_t_minus_1) * dW_2t
        )
        carry = nu_2_t
        return carry, carry

    _, vec_nu_2 = jax.lax.scan(_calc_nu_2_t, init=nu_2_init, xs=dW_2[1:])
    vec_nu_2 = jnp.insert(arr=vec_nu_2, obj=0, values=nu_2_init, axis=0)

    return vec_nu_2


@jit
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


@jit
def calc_nu(
    beta,
    vec_nu_1: jpt.Float[jpt.Array, "ts.size"],
    vec_nu_2: jpt.Float[jpt.Array, "ts.size"],
) -> jpt.Float[jpt.Array, "ts.size"]:
    """
    Calculate \\nu_t = s-exp( \\beta_0 + \\beta_1\\nu_{1t} + \\beta_2\\nu_{2t})
    """
    val = beta[0] + beta[1] * vec_nu_1 + beta[2] * vec_nu_2
    return spline_exponential(val)


@jit
def calc_sigma_u(
    tt: jpt.Float,
    A: jpt.Float,
    B: jpt.Float,
    C: jpt.Float,
    a: jpt.Float,
    b: jpt.Float,
    t_end: jpt.Float = 1.0,
) -> jpt.Float:
    """
    Diurnal U-shape function \\sigma_{ut} = C + A e^{-at} + B e^{-b(1 - t)} over [0, T]
    """
    return C + A * jnp.exp(-a * tt) + B * jnp.exp(-b * (t_end - tt))


# @jit
def calc_dlogS_oneday(
    ts: TimeSequence,
    delta_t: jpt.Float,
    mu: jpt.Float,
    rho: jpt.Array,
    alpha: jpt.Array,
    phi: jpt.Float,
    beta: jpt.Array,
    dict_params_sigma_u: tp.Dict[str, jpt.Float],
    dW: SimulationPathOneDay3,
    init_S: jpt.Float,
    init_nu: jpt.Float,
) -> tp.Tuple[
    SimulationPathOneDay,  # vec_nu_1
    SimulationPathOneDay,  # vec_nu_2
    SimulationPathOneDay,  # vec_logS
]:
    """
    Simulate prices for a single day.
    """

    def _drift():
        return mu

    def _diffusion_1(sigma_u_t, nu_t):
        """
        Diffusion term \\rho_1\\sigma_{ut}\\nu_t
        """
        return rho[0] * sigma_u_t * nu_t

    def _diffusion_2(sigma_u_t, nu_t):
        """
        Diffusion term \\rho_2\\sigma_{ut}\\nu_t
        """
        return rho[1] * sigma_u_t * nu_t

    def _diffusion_3(sigma_u_t, nu_t):
        """
        Diffusion term \\sqrt{1 - \\rho_1^2 - \\rho_2^2}\\sigma_{ut}\\nu_t
        """
        return jnp.sqrt(1 - rho[1] ** 2 - rho[2] ** 2) * sigma_u_t * nu_t

    @jit
    def _calc_logS_t(carry, xs):
        dW_t, sigma_u_t_minus_1, nu_t_minus_1 = xs
        logS_t_minus_1 = carry

        logS_t = (
            logS_t_minus_1
            + _drift() * delta_t
            + _diffusion_1(sigma_u_t_minus_1, nu_t_minus_1) * dW_t[0]
            + _diffusion_2(sigma_u_t_minus_1, nu_t_minus_1) * dW_t[1]
            + _diffusion_3(sigma_u_t_minus_1, nu_t_minus_1) * dW_t[2]
        )
        carry = logS_t
        return carry, carry

    # Compute \\nu_t's
    vec_nu_1 = calc_dnu_1(
        delta_t=delta_t,
        alpha_1=alpha[0],
        dW_1=dW[:, 0],
        nu_1_init=init_nu[0],
    )
    vec_nu_2 = calc_dnu_2(
        delta_t=delta_t,
        alpha_2=alpha[1],
        phi=phi,
        dW_2=dW[:, 1],
        nu_2_init=init_nu[1],
    )
    vec_nu = calc_nu(
        beta=beta,
        vec_nu_1=vec_nu_1,
        vec_nu_2=vec_nu_2,
    )

    # Compute \\sigma_{ut}'s
    vec_sigma_u = calc_sigma_u(ts, **dict_params_sigma_u)
    logS_init = jnp.log(init_S)
    xs = (dW[1:], vec_sigma_u[:-1], vec_nu[:-1])
    _, vec_logS = jax.lax.scan(_calc_logS_t, init=logS_init, xs=xs)
    vec_logS = jnp.insert(vec_logS, 0, logS_init, axis=0)

    return vec_nu_1, vec_nu_2, vec_logS


def calc_dlogS_all_days(
    params: ParamsTwoFactorSV,
    all_days_dW: SimulationPathAllDays3,
) -> tp.Tuple[jpt.Array, jpt.Array, jpt.Array]:
    """
    Simulate log-prices across all days.
    """
    # Split Brownian shocks into separate days
    lst_split_dW = jnp.split(all_days_dW, params.total_num_days)

    dict_params_sigma_u = {
        "A": params.A,
        "B": params.B,
        "C": params.C,
        "a": params.a,
        "b": params.b,
    }

    lst_vec_logS = []
    lst_vec_nu_1 = []
    lst_vec_nu_2 = []
    init_S = params.init_price
    init_nu = params.init_nu
    for day in range(params.total_num_days):
        dW = lst_split_dW[day]
        vec_nu_1, vec_nu_2, vec_logS = calc_dlogS_oneday(
            ts=params.ts,
            delta_t=params.delta_t,
            mu=params.mu,
            rho=params.rho,
            alpha=params.alpha,
            phi=params.phi,
            beta=params.beta,
            dict_params_sigma_u=dict_params_sigma_u,
            dW=dW,
            init_S=init_S,
            init_nu=init_nu,
        )

        # Append
        lst_vec_logS.append(vec_logS)
        lst_vec_nu_1.append(vec_nu_1)
        lst_vec_nu_2.append(vec_nu_2)

        # Reset initial conditions
        init_S = jnp.exp(vec_logS[-1])
        init_nu = jnp.array([vec_nu_1[-1], vec_nu_2[-1]])

    log_price = jnp.concat(lst_vec_logS)
    vol_factor_1 = jnp.concat(lst_vec_nu_1)
    vol_factor_2 = jnp.concat(lst_vec_nu_2)
    return vol_factor_1, vol_factor_2, log_price


def simulate_dlogS_all_epochs(
    params: ParamsTwoFactorSV,
) -> SimulationPathAllDaysAllEpochs:
    """
    Simulate log-prices across all days and all epochs.
    """
    # Simulate Brownian motions across all epochs and all days
    all_dW = calc_dW(
        key=params.key,
        delta_t=params.delta_t,
        num_discr=params.total_num_days * params.num_time_units_per_day,
        num_replications=params.total_num_epochs,
        d=params.d,
    )

    @jit
    def _func(all_days_dW):
        _, _, log_price = calc_dlogS_all_days(
            params=params,
            all_days_dW=all_days_dW,
        )
        return log_price

    res = jax.vmap(_func, in_axes=[0])(all_dW)
    return res


if __name__ == "__main__":
    params = ParamsTwoFactorSV()

    t0 = time.time()
    hi = simulate_dlogS_all_epochs(
        params=params,
    )
    t1 = time.time()
    tot_time = round(t1 - t0, 2)
    print(f"Simulation time {tot_time} sec")

    for epoch in range(params.total_num_epochs):
        plt.plot(hi[epoch, :])
    plt.show()

    breakpoint()

    # # Simulate across all epochs
    # lst_price_epochs = []
    # lst_vol_factor_1_epochs = []
    # lst_vol_factor_2_epochs = []
    # for epoch in range(num_epochs):
    #     all_days_dW = all_dW[epoch, :, :]
    #     vol_factor_1, vol_factor_2, price = calc_dlogS_all_days(
    #         total_num_days=total_num_days,
    #         delta_t=delta_t,
    #         params=params,
    #         all_days_dW=all_days_dW,
    #         init_price=init_price,
    #         init_nu=init_nu,
    #     )
    #
    #     lst_price_epochs.append(price)
    #     lst_vol_factor_1_epochs.append(vol_factor_1)
    #     lst_vol_factor_2_epochs.append(vol_factor_2)

    # for epoch in range(num_epochs):
    #     plt.plot(lst_price_epochs[epoch])
    # plt.show()
    #
    # for epoch in range(num_epochs):
    #     plt.plot(lst_vol_factor_1_epochs[epoch])
    # plt.show()
    #
    # for epoch in range(num_epochs):
    #     plt.plot(lst_vol_factor_2_epochs[epoch])
    # plt.show()
    #
