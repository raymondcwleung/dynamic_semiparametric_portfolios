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

# import tensorflow as tf
import pandas as pd


current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=f"logs/{current_time}_gas_var.log",
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

import utils

# HACK:
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
#jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod
jax.config.update("jax_debug_nans", True)  # Should disable in full prod

NUM_GAS_PARAMS = 2

# def scoring_function(r: jpt.Float, v: jpt.Float, alpha: jpt.Float) -> jpt.Float:
#     """
#     Scoring function S(r, v ; \\alpha)
#     = (\\ind(r \\le v) - \\alpha)(v - \\alpha)
#     """
#     indicator = utils.indicator
#
#     score = (indicator(r - v) - alpha) - (v - alpha)
#     return score


def scoring_function(
    r: jpt.Float, v: jpt.Float, e: jpt.Float, alpha: jpt.Float
) -> jpt.Float:
    """
    Scoring function S(r, v, e ; \\alpha)
    """
    indicator = utils.indicator
    log = jnp.log

    score = -(1 / (alpha * e)) * indicator(r - v) * (r - v) + (v / e) + log(-e) - 1
    return score


# def _one_factor_gas(
#     theta: jpt.Float[jpt.Array, "NUM_GAS_PARAMS"],
#     theta_X: jpt.Float[jpt.Array, "dimX"],
#     alpha: jpt.Float,
#     return_t_minus_one: jpt.Float,
#     kappa_t_minus_one: jpt.Float,
#     vec_X_t_minus_one: jpt.Float[jpt.Array, "dimX"],
# ) -> jpt.Float:
#     """
#     Return the GAS(1,1) model \\kappa_t
#     """
#     indicator = utils.indicator
#     exp = jnp.exp
#
#     kappa_t = (
#         theta[1]
#         + theta[2] * kappa_t_minus_one
#         + jnp.inner(theta_X, vec_X_t_minus_one)
#         + theta[3]
#         * (indicator(return_t_minus_one - theta[0] * exp(kappa_t_minus_one)) - alpha)
#     )
#     return kappa_t


def _one_factor_gas(
    a: jpt.Float,
    b: jpt.Float,
    theta: jpt.Float[jpt.Array, "NUM_GAS_PARAMS"],
    theta_X: jpt.Float[jpt.Array, "dimX"],
    alpha: jpt.Float,
    return_t_minus_one: jpt.Float,
    kappa_t_minus_one: jpt.Float,
    vec_X_t_minus_one: jpt.Float[jpt.Array, "dimX"],
) -> jpt.Float:
    """
    Return the GAS(1,1) model \\kappa_t

    Parameters are such that theta := [\\beta, \\gamma]
    as per Patton, Ziegel and Chen (2019); note the fixing of \\omega = 0
    for identification. Constraints are such that b < a < 0.
    """
    indicator = utils.indicator
    exp = jnp.exp

    kappa_t = theta[0] * kappa_t_minus_one + theta[1] / (b * exp(kappa_t_minus_one)) * (
        (1 / alpha)
        * indicator(return_t_minus_one - a * exp(kappa_t_minus_one))
        * return_t_minus_one
        - b * exp(kappa_t_minus_one)
    )
    return kappa_t


def one_factor_gas(
    a: jpt.Float,
    b: jpt.Float,
    theta: jpt.Float[jpt.Array, "NUM_GAS_PARAMS"],
    data_returns: jpt.Float[jpt.Array, "num_sample"],
    alpha: jpt.Float,
    kappa_init_t0: jpt.Float,
) -> jpt.Float[jpt.Array, "num_sample"]:
    """
    Return the full-sample GAS(1,1) model \\kappa_t for all t.
    """
    num_sample = jnp.size(data_returns)

    # Nullify the exogenous variables
    theta_X = jnp.array([0.0])
    vec_X_t_minus_one = jnp.array([0.0])

    vec_kappa = jnp.empty(shape=(num_sample,))
    vec_kappa = vec_kappa.at[0].set(kappa_init_t0)

    def _body_fun(tt, vec_kappa):
        kappa_t_minus_one = vec_kappa[tt - 1]
        return_t_minus_one = data_returns[tt - 1]

        kappa_t = _one_factor_gas(
            a=a,
            b=b,
            theta=theta,
            theta_X=theta_X,
            alpha=alpha,
            return_t_minus_one=return_t_minus_one,
            kappa_t_minus_one=kappa_t_minus_one,
            vec_X_t_minus_one=vec_X_t_minus_one,
        )
        vec_kappa = vec_kappa.at[tt].set(kappa_t)
        return vec_kappa

    # Compute \kappa_t's
    vec_kappa = jax.lax.fori_loop(
        lower=1, upper=num_sample, body_fun=_body_fun, init_val=vec_kappa
    )
    return vec_kappa


def _calc_VaR(
    a : jpt.Float,
    vec_kappa: jpt.Float[jpt.Array, "num_sample"],
) -> jpt.Float[jpt.Array, "num_sample"]:
    """
    Mapping from the GAS scores \\kappa_t to the VaR v_t
    """
    vec_VaR = a * jnp.exp(vec_kappa)
    return vec_VaR


def _calc_expected_shortfall(
    b : jpt.Float,
    vec_kappa: jpt.Float[jpt.Array, "num_sample"],
):
    """
    Mapping from the GAS scores \\kappa_t to the ES e_t
    """
    vec_ES = b * jnp.exp(vec_kappa)
    return vec_ES




def gas_VaR_ES(
    a: jpt.Float,
    b: jpt.Float,
    theta: jpt.Float[jpt.Array, "NUM_GAS_PARAMS"],
    data_returns: jpt.Float[jpt.Array, "num_sample"],
    alpha: jpt.Float,
    var_init_t0: jpt.Float,
) -> tp.Tuple[jpt.Float[jpt.Array, "num_sample"], jpt.Float[jpt.Array, "num_sample"]]:
    """
    Return the time-varying (v_t, e_t) where v_t = a \\exp{\\kappa_t} is the VaR,
    and e_t = b \\exp{\\kappa_t} is the expected shortfall over the full-sample.

    NOTE: We extract the initial t = 0 condition \\kappa_0 only from the initial
    VaR v_0 (i.e. and not from the expected shortfall e_0).
    """
    kappa_init_t0 = jnp.log(var_init_t0 / a)

    vec_kappa = one_factor_gas(
        a=a,
        b=b,
        theta=theta,
        data_returns=data_returns,
        alpha=alpha,
        kappa_init_t0=kappa_init_t0,
    )
    vec_VaR = _calc_VaR(a = a, vec_kappa=vec_kappa)
    vec_ES = _calc_expected_shortfall(b = b, vec_kappa=vec_kappa)

    return vec_VaR, vec_ES


def sample_score(
    a: jpt.Float,
    b: jpt.Float,
    theta: jpt.Float[jpt.Array, "NUM_GAS_PARAMS"],
    data_returns: jpt.Float[jpt.Array, "num_sample"],
    alpha: jpt.Float,
    VaR_init_t0: jpt.Float,
) -> jpt.Float:
    """
    Given returns \\{ R_t \\}, compute the sample moment
    \\frac{1}{T} \\sum_t S(R_t, v_t, e_t ; \\alpha), where
    v_t is the GAS-VaR model and e_t is the GAS-ES model.
    """
    # Compute VaR v_t's and ES e_t's
    vec_VaR, vec_ES = gas_VaR_ES(
        a=a,
        b=b,
        theta=theta,
        data_returns=data_returns,
        alpha=alpha,
        var_init_t0=VaR_init_t0,
    )

    # Evaluate at the score functions
    vec_scores = scoring_function(r=data_returns, v=vec_VaR, e=vec_ES, alpha=alpha)

    # Compute the sample scores
    sample_score_val = jnp.mean(vec_scores)

    # Return the negaive of the sample scores
    return -1 * sample_score_val


if __name__ == "__main__":
    fn = "./data/yjdeqke4kocqonq9.csv"  # S&P 500 daily returns
    df = pd.read_csv(fn)

    data_returns = jnp.array(df["vwretd"].values)

    # Parameters are such that theta := [\\beta, \\gamma]
    VaR_init_t0 = -2.95
    a = -1.164
    b = -1.757
    theta = jnp.array([0.995, 0.007])
    alpha = 0.01
    hi = sample_score(
        a=a,
        b=b,
        theta=theta,
        data_returns=data_returns,
        alpha=alpha,
        VaR_init_t0=VaR_init_t0,
    )


    solver = optax.adam

