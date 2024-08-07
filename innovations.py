import jax
from jax._src.random import KeyArrayLike
import jax.numpy as jnp
import jax.scipy as jscipy
import jax.scipy.optimize
from jax import grad, jit, vmap
from jax import random
import jax.test_util

import typing as tp
import jaxtyping as jpt

import chex
from matplotlib._api import suppress_matplotlib_deprecation_warning

import numpy.typing as npt


import os
import logging
from pathlib import Path
import pickle

import itertools
from functools import partial

from dataclasses import dataclass
import jax.scipy as jscipy

# import optax
# import jaxopt

import numpy as np
import scipy
import matplotlib.pyplot as plt

import utils

# HACK:
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
# jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod
# jax.config.update("jax_debug_nans", True)  # Should disable in full prod


logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename="sgt.log",
#     datefmt="%Y-%m-%d %I:%M:%S %p",
#     level=logging.INFO,
#     format="%(levelname)s | %(asctime)s | %(message)s",
#     filemode="w",
# )
#
#
RUN_TIMEVARYING_SGT_SIMULATIONS = True
NUM_LBDA_TVPARAMS = 3
NUM_P0_TVPARAMS = 3
NUM_Q0_TVPARAMS = 3


@chex.dataclass(kw_only=True)
class ParamsZ:
    """
    Generic placeholder for the parent class of parameters for an
    innovations process z_t
    """


@chex.dataclass
class ParamsZGaussian(ParamsZ):
    mean: jpt.Float[jpt.Array, "dim"]
    cov: jpt.Float[jpt.Array, "dim dim"]


@chex.dataclass
class ParamsZSgt(ParamsZ):
    """
    Time-varying parameters of innovations process z_t
    """

    mat_lbda_tvparams: jpt.Float[jpt.Array, "NUM_LBDA_TVPARAMS dim"]
    mat_p0_tvparams: jpt.Float[jpt.Array, "NUM_P0_TVPARAMS dim"]
    mat_q0_tvparams: jpt.Float[jpt.Array, "NUM_Q0_TVPARAMS dim"]

    # num_lbda_tvparams: int = NUM_LBDA_TVPARAMS
    # num_p0_tvparams: int = NUM_P0_TVPARAMS
    # num_q0_tvparams: int = NUM_Q0_TVPARAMS

    def check_constraints(self) -> jpt.Bool:
        # Constraints on the data generating process of \lambda_t
        valid_constraints_lbda = jax.lax.select(
            pred=jnp.any(self.mat_lbda_tvparams[0, :] > 0),
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )

        # Constraints on the data generating process of p_t
        valid_constraints_p0 = jax.lax.select(
            pred=jnp.any(self.mat_p0_tvparams[0, :] > 0),
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )

        # Constraints on the data generating process of q_t
        valid_constraints_q0 = jax.lax.select(
            pred=jnp.any(self.mat_q0_tvparams[0, :] > 0),
            on_true=jnp.array(True),
            on_false=jnp.array(False),
        )

        vec_constraints = jnp.array(
            [valid_constraints_lbda, valid_constraints_p0, valid_constraints_q0]
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
class InitTimeConditionZSgt:
    """
    Initial conditions related to the innovation process z_t.
    NOTE: This is primarily only used for simulating data and
    not on real data.
    """

    vec_z_init_t0: jpt.Float[jpt.Array, "dim"]
    vec_lbda_init_t0: jpt.Float[jpt.Array, "dim"]
    vec_p0_init_t0: jpt.Float[jpt.Array, "dim"]
    vec_q0_init_t0: jpt.Float[jpt.Array, "dim"]

    # def __post_init__(self):
    #     # Parameter constraint checks on \lambda_0
    #     if jnp.any(self.vec_lbda_init_t0 <= -1) or jnp.any(self.vec_lbda_init_t0 >= 1):
    #         raise ValueError(
    #             "For all t, the process \\lambda_t (notably at t = 0) must be in (-1, 1)."
    #         )
    #
    #     # Parameter constraint checks on q
    #     if jnp.any(self.vec_p0_init_t0 <= 0):
    #         raise ValueError(
    #             "For all t, the process p_t (notably at t = 0) must be > 0."
    #         )
    #
    #     # Parameter constraint checks on q
    #     if jnp.any(self.vec_q0_init_t0 <= 0):
    #         raise ValueError(
    #             "For all t, the process q_t (notably at t = 0) must be > 0."
    #         )


@chex.dataclass
class SimulatedInnovations:
    """
    Parent class for all simulated innovations
    """

    # Number of time samples
    num_sample: int

    # Simulated innovations z_t data
    data_mat_z: jpt.Float[jpt.Array, "num_sample dim"]


@chex.dataclass
class SimulatedGaussianInnovations(SimulatedInnovations):
    params_z: ParamsZGaussian


@chex.dataclass
class SimulatedSGTInnovations(SimulatedInnovations):
    """
    Data class for keeping track of the parameters and data
    of the innovation SGT z_t
    """

    ################################################################
    ## Parameters
    ################################################################
    params_z_sgt: ParamsZSgt

    ################################################################
    ## Initial t = 0 conditions for the time-varying parameters
    ################################################################
    inittimecond_z_sgt: InitTimeConditionZSgt

    ################################################################
    ## Simulated data
    ################################################################
    # Time-varying parameters related to z_t
    data_mat_lbda: jpt.Float[jpt.Array, "num_sample dim"]
    data_mat_p0: jpt.Float[jpt.Array, "num_sample dim"]
    data_mat_q0: jpt.Float[jpt.Array, "num_sample dim"]


@partial(jax.jit, static_argnames=["mu", "sigma", "mean_cent", "var_adj"])
def pdf_sgt(z, lbda, p0, q0, mu=0.0, sigma=1.0, mean_cent=True, var_adj=True):
    """
    Univariate SGT density
    """
    power = jnp.power
    sqrt = jnp.sqrt
    sign = jnp.sign
    abs = jnp.abs
    beta = jscipy.special.beta

    if var_adj:
        v = q0 ** (1 / p0) * sqrt(
            (1 + 3 * lbda**2) * (beta(3 / p0, q0 - 2 / p0) / beta(1 / p0, q0))
            - 4 * lbda**2 * (beta(2 / p0, q0 - 1 / p0) / beta(1 / p0, q0)) ** 2
        )
        sigma = sigma / v

    if mean_cent:
        m = (2 * sigma * lbda * q0 ** (1 / p0) * beta(2 / p0, q0 - 1 / p0)) / beta(
            1 / p0, q0
        )
        z = z + m

    density = p0 / (
        2
        * sigma
        * q0 ** (1 / p0)
        * beta(1 / p0, q0)
        * power(
            1
            + abs(z - mu) ** p0 / (q0 * sigma**p0 * power(1 + lbda * sign(z - mu), p0)),
            1 / p0 + q0,
        )
    )
    return density


def pdf_mvar_indp_sgt(
    x: jpt.Float[jpt.Array, "dim"],
    vec_lbda: jpt.Float[jpt.Array, "dim"],
    vec_p0: jpt.Float[jpt.Array, "dim"],
    vec_q0: jpt.Float[jpt.Array, "dim"],
):
    """
    Let X_1 \\sim SGT(\\theta_1), \\ldots, X_d \\sim SGT(\\theta_d) be
    independent. Construct the random vector X = (X_1, \\ldots, X_d).
    Return the density of X.
    """
    _func = vmap(pdf_sgt, in_axes=[0, 0, 0, 0])
    vec_pdf = _func(x, vec_lbda, vec_p0, vec_q0)
    return vec_pdf


def loglik_mvar_indp_sgt(
    data: jpt.Array,
    vec_lbda: jpt.Array,
    vec_p0: jpt.Array,
    vec_q0: jpt.Array,
    neg_loglik: bool,
):
    """
    (Negative) of the log-likelihood function of a vector of
    independent SGT random variables.
    """
    _func = vmap(pdf_mvar_indp_sgt, in_axes=[0, None, None, None])

    summands = _func(data, vec_lbda, vec_p0, vec_q0)
    loglik_summands = jnp.log(summands)
    loglik = loglik_summands.sum()

    if neg_loglik:
        loglik = -1.0 * loglik
    return loglik


def loglik_mvar_timevarying_sgt(
    data: jpt.Float[jpt.Array, "num_sample dim"],
    mat_lbda: jpt.Float[jpt.Array, "num_sample dim"],
    mat_p0: jpt.Float[jpt.Array, "num_sample dim"],
    mat_q0: jpt.Float[jpt.Array, "num_sample dim"],
    neg_loglik: bool,
) -> jpt.Float:
    """
    (Negative) of the log-likelihood function of a vector of
    independent SGT random variables.
    """
    num_sample = data.shape[0]
    _func = vmap(pdf_mvar_indp_sgt, in_axes=[0, 0, 0, 0])

    summands = _func(data, mat_lbda, mat_p0, mat_q0)
    summands = jnp.prod(summands, axis=1)

    # Ensure the individual pdf values in the likelihood
    # function are correctly sized
    if jnp.size(summands) != num_sample:
        raise ValueError(
            "Individual multiplicands in the likelihood function are not correctly sized."
        )

    loglik_summands = jnp.log(summands)
    loglik = loglik_summands.sum()

    if neg_loglik:
        loglik = -1.0 * loglik
    return loglik


def quantile_sgt(
    prob: jpt.Float[jpt.Array, "?num_sample"],
    lbda: jpt.Float[jpt.Array, "?dim"],
    p0: jpt.Float[jpt.Array, "?dim"],
    q0: jpt.Float[jpt.Array, "?dim"],
    mu: jpt.Float = 0.0,
    sigma: jpt.Float = 1.0,
    mean_cent: bool = True,
    var_adj: bool = True,
    use_jax: bool = True,
) -> jpt.Float:
    """
    Univariate SGT quantile
    """
    # PERF: Slow performance here because of the
    # use of scipy.stats.beta.ppf. In particular,
    # the JAX counterpart to stats.beta.ppf
    # has NOT been implemented. Hence, we must
    # resort to the slow scipy.stats.beta.ppf.
    beta_quantile = scipy.stats.beta.ppf

    if use_jax:
        sqrt = jnp.sqrt
        beta = jscipy.special.beta
    else:
        sqrt = np.sqrt
        beta = scipy.special.beta

    v = q0 ** (1 / p0) * sqrt(
        (1 + 3 * lbda**2) * (beta(3 / p0, q0 - 2 / p0) / beta(1 / p0, q0))
        - 4 * lbda**2 * (beta(2 / p0, q0 - 1 / p0) / beta(1 / p0, q0)) ** 2
    )
    if var_adj:
        sigma = sigma / v

    lam = lbda

    flip = prob > (1 - lbda) / 2
    # if flip:
    #     prob = 1 - prob
    #     lam = -1 * lam
    prob = flip * (1 - prob) + (1 - flip) * prob
    lam = flip * (-1 * lam) + (1 - flip) * lam

    out = (
        sigma
        * (lam - 1)
        * (
            1 / (q0 * beta_quantile(q=1 - 2 * prob / (1 - lam), a=1 / p0, b=q0))
            - 1 / q0
        )
        ** (-1 / p0)
    )

    # if flip:
    #     out = -out
    out = flip * (-1 * out) + (1 - flip) * out

    out = out + mu

    if mean_cent:
        m = (2 * sigma * lbda * q0 ** (1 / p0) * beta(2 / p0, q0 - 1 / p0)) / beta(
            1 / p0, q0
        )
        out = out - m

    return out


def sample_mvar_sgt(
    key,
    num_sample: int,
    vec_lbda: jpt.ArrayLike,
    vec_p0: jpt.ArrayLike,
    vec_q0: jpt.ArrayLike,
    mu: float = 0.0,
    sigma: float = 1.0,
    num_cores: int = 1,
) -> jpt.Array:
    """
    Generate samples of SGT random vectors by
    inverse transform sampling.

    NOTE: This function does not use JAX.
    """
    import multiprocessing as mp

    dim = jnp.size(vec_lbda)

    vec_mu = np.repeat(mu, dim)
    vec_sigma = np.repeat(sigma, dim)

    subkeys = random.split(key, num_sample)
    unif_data = jax.random.uniform(key=key, shape=(num_sample, dim))
    np_unif_data = np.array(unif_data)

    args = itertools.product(
        np_unif_data, [vec_lbda], [vec_p0], [vec_q0], [vec_mu], [vec_sigma]
    )
    with mp.Pool(num_cores) as pool:
        lst_data = pool.starmap(quantile_sgt, args)
    data = jnp.array(lst_data)

    return data


def _generate_mvar_sgt_param_init_guesses(
    key,
    dim: int,
    num_trials: int,
    lbda_a: float,
    lbda_b: float,
    p0_a: float,
    p0_b: float,
    q0_a: float,
    q0_b: float,
):
    """
    Generate random initial guesses for the MLE estimation
    of the SGT density. In particular, the parameters are randomly
    generated as follows. Let a, b be two scalars. If U \\sim Uniform[0,1]
    then the parameter in question is generated as a + b U.
    """
    key, subkey = random.split(key)
    lbda_guesses = lbda_a + lbda_b * jax.random.uniform(key, shape=(num_trials, dim))
    lbda_guesses = np.array(lbda_guesses)

    key, subkey = random.split(key)
    p0_guesses = p0_a + p0_b * jax.random.uniform(key, shape=(num_trials, dim))
    p0_guesses = np.array(p0_guesses)

    key, subkey = random.split(key)
    q0_guesses = q0_a + q0_b * jax.random.uniform(key, shape=(num_trials, dim))
    q0_guesses = np.array(q0_guesses)

    lst_x0 = list(itertools.product(lbda_guesses, p0_guesses, q0_guesses))
    return lst_x0


def mvar_sgt_objfun(x, data, neg_loglik: bool = True):
    """
    (Negative) of the log-likelihood of the a vector of
    independent SGT random variables.
    """
    dim = data.shape[1]

    vec_lbda = x[0:dim]
    vec_p0 = x[dim : 2 * dim]
    vec_q0 = x[2 * dim :]

    return loglik_mvar_indp_sgt(
        data=data,
        vec_lbda=vec_lbda,
        vec_p0=vec_p0,
        vec_q0=vec_q0,
        neg_loglik=neg_loglik,
    )


def mle_mvar_sgt(
    key,
    data: jpt.Array,
    num_trials: int,
    lst_scale_lbda: list[float] = [-0.25, 0.5],
    lst_scale_p0: list[float] = [2, 5],
    lst_scale_q0: list[float] = [2, 5],
    options: dict = {"maxiter": 1000, "gtol": 1e-3},
):
    """
    MLE for the SGT density.

    Parameters
    ----------
    num_trials: Number of random numbers to generate for
        each parameter.
    """
    logger.info("Begin MLE for SGT")

    dim = data.shape[1]

    # Generate the random initial conditions
    lst_x0 = _generate_mvar_sgt_param_init_guesses(
        key=key,
        dim=dim,
        num_trials=num_trials,
        lbda_a=lst_scale_lbda[0],
        lbda_b=lst_scale_lbda[1],
        p0_a=lst_scale_p0[0],
        p0_b=lst_scale_p0[1],
        q0_a=lst_scale_q0[0],
        q0_b=lst_scale_q0[1],
    )

    # Multi-start local method
    optres = None
    for ii in range(len(lst_x0)):
        print(f"Iteration num {ii}/{len(lst_x0)}")

        x0 = jnp.array(lst_x0[ii])
        x0 = jnp.ravel(x0)
        try:
            nowres = jscipy.optimize.minimize(
                mvar_sgt_objfun, x0=x0, method="BFGS", args=(data,), options=options
            )

            if nowres.success and (optres is None or optres.fun > nowres.fun):
                optres = nowres
            else:
                logger.info(f"No solution at iteration {ii}. x0 = {x0}")

        except FloatingPointError as e:
            logger.warning(f"Iteration {ii} encountered FloatingPointError. x0 = {x0}")
            logger.warning(str(e))
            continue

    if (optres is None) or (optres.success is False):
        logger.critical("No solution found!")
    else:
        logger.info("Done! Complete MLE for SGT.")

    return optres


def _time_varying_lbda_params(
    theta: jpt.Float[jpt.Array, "NUM_LBDA_TVPARAMS"],
    lbda_t_minus_1: jpt.Float[jpt.Array, "#dim"],
    z_t_minus_1: jpt.Float[jpt.Array, "#dim"],
) -> jpt.Array:
    """
    Time varying dynamics of the \\lambda parameter.
    """
    tanh = jnp.tanh
    arctanh = jnp.arctanh
    negative_part = utils.negative_part
    positive_part = utils.positive_part
    indicator = utils.indicator

    _rhs = (
        theta[0]
        + negative_part(theta[1]) * z_t_minus_1 * indicator(z_t_minus_1)
        + positive_part(theta[1]) * z_t_minus_1 * (1 - indicator(z_t_minus_1))
    )

    lbda_t = tanh(_rhs + theta[2] * arctanh(lbda_t_minus_1))
    return lbda_t


def _time_varying_pq_params(
    theta: jpt.Float[jpt.Array, "?num_pq_tvparams"],
    param_t_minus_1: jpt.Float[jpt.Array, "?num_pq_tvparams"],
    z_t_minus_1: jpt.Float[jpt.Array, "dim"],
    theta_bar: float = 2.0,
) -> jpt.Array:
    """
    Time varying dynamics of the p or q parameters.
    """
    abs = jnp.abs
    exp = jnp.exp
    log = jnp.log
    negative_part = utils.negative_part
    positive_part = utils.positive_part
    indicator = utils.indicator

    try:
        # Define the transition terms on the RHS
        _rhs = (
            log(1 + theta[0])
            + negative_part(theta[1]) * abs(z_t_minus_1) * indicator(z_t_minus_1)
            + positive_part(theta[1]) * abs(z_t_minus_1) * (1 - indicator(z_t_minus_1))
        )

        # param_t = theta_bar + exp(_rhs + theta[2] * log(param_t_minus_1 - theta_bar))
        param_t = exp(_rhs + theta[2] * log(param_t_minus_1))

        # TODO: Think about this spec?
        param_t = jnp.max(jnp.array([theta_bar, param_t]))

    except FloatingPointError as e:
        # Floating point error most likely in the log(.) calculation
        logger.info(str(e))
        logger.debug(str(e))
        param_t = param_t_minus_1

    except Exception as e:
        logger.debug(str(e))
        param_t = param_t_minus_1

    return param_t


def sample_mvar_gaussian(
    key: KeyArrayLike, num_sample: int, params_z_gaussian_true: ParamsZGaussian
) -> SimulatedGaussianInnovations:
    """
    Generate samples of iid standard multivariate
    Gaussian vectors
    """
    dim = jnp.shape(params_z_gaussian_true.cov)[0]

    mean = params_z_gaussian_true.mean
    cov = params_z_gaussian_true.cov

    data_mat_z = jax.random.multivariate_normal(
        key=key, mean=mean, cov=cov, shape=(num_sample,)
    )
    siminnov = SimulatedGaussianInnovations(
        num_sample=num_sample, data_mat_z=data_mat_z, params_z=params_z_gaussian_true
    )

    return siminnov


@jit
def loglik_std_gaussian(data: jpt.Float[jpt.Array, "num_sample data"]) -> jpt.Float:
    """
    Log-likleihood function of a vector of
    standardized independent Gaussian random vectors
    """
    # Dropping constants
    loglik = -0.5 * jnp.sum(jnp.sum(data * data, axis=1))
    return loglik


def sample_mvar_timevarying_sgt(
    key: KeyArrayLike,
    num_sample: int,
    params_z_sgt_true: ParamsZSgt,
    inittimecond_z_sgt: InitTimeConditionZSgt,
    save_path: None | os.PathLike,
) -> SimulatedSGTInnovations:
    """
    Generate samples of time-varying SGT random
    vectors by inverse transform sampling.

    NOTE: This function does NOT use JAX.
    """
    dim = jnp.shape(params_z_sgt_true.mat_lbda_tvparams)[1]

    # Independent Uniform(0,1) random variables
    unif_data = jax.random.uniform(key=key, shape=(num_sample, dim))

    # Init
    mat_lbda = jnp.empty(shape=(num_sample, dim))
    mat_p0 = jnp.empty(shape=(num_sample, dim))
    mat_q0 = jnp.empty(shape=(num_sample, dim))
    mat_z = jnp.empty(shape=(num_sample, dim))

    mat_lbda = mat_lbda.at[0].set(inittimecond_z_sgt.vec_lbda_init_t0)
    mat_p0 = mat_p0.at[0].set(inittimecond_z_sgt.vec_p0_init_t0)
    mat_q0 = mat_p0.at[0].set(inittimecond_z_sgt.vec_q0_init_t0)
    mat_z = mat_z.at[0].set(inittimecond_z_sgt.vec_z_init_t0)

    # Setup vmap functions
    _func_lbda = vmap(_time_varying_lbda_params, in_axes=[1, 0, 0])
    _func_pq = vmap(_time_varying_pq_params, in_axes=[1, 0, 0])

    def _body_fun(tt, carry):
        """
        Convenient function for mapping t - 1 quantities to t quantities
        """
        mat_lbda, mat_p0, mat_q0, mat_z = carry

        # Compute \lambda_t
        vec_lbda_t = _func_lbda(
            params_z_sgt_true.mat_lbda_tvparams, mat_lbda[tt - 1, :], mat_z[tt - 1, :]
        )

        # Compute p_t
        vec_p0_t = _func_pq(
            params_z_sgt_true.mat_p0_tvparams, mat_p0[tt - 1, :], mat_z[tt - 1, :]
        )

        # Compute q_t
        vec_q0_t = _func_pq(
            params_z_sgt_true.mat_q0_tvparams, mat_q0[tt - 1, :], mat_z[tt - 1, :]
        )

        # Compute z_t
        vec_z_t = quantile_sgt(unif_data[tt, :], vec_lbda_t, vec_p0_t, vec_q0_t)

        # Bookkeeping
        mat_lbda = mat_lbda.at[tt].set(vec_lbda_t)
        mat_p0 = mat_p0.at[tt].set(vec_p0_t)
        mat_q0 = mat_q0.at[tt].set(vec_q0_t)
        mat_z = mat_z.at[tt].set(vec_z_t)

        return mat_lbda, mat_p0, mat_q0, mat_z

    logger.debug(f"Begin time-varying SGT simulation")

    carry = (mat_lbda, mat_p0, mat_q0, mat_z)
    for tt in range(1, num_sample):
        carry = _body_fun(tt, carry)

        if tt % 100 == 0:
            logger.info(f"... complete iteration {tt}/{num_sample}")

    data_mat_lbda, data_mat_p0, data_mat_q0, data_mat_z = carry

    logger.debug(f"Done time-varying SGT simulation")

    siminnov = SimulatedSGTInnovations(
        num_sample=num_sample,
        params_z_sgt=params_z_sgt_true,
        inittimecond_z_sgt=inittimecond_z_sgt,
        data_mat_lbda=data_mat_lbda,
        data_mat_p0=data_mat_p0,
        data_mat_q0=data_mat_q0,
        data_mat_z=data_mat_z,
    )

    # Save
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(siminnov, f)
        logger.info(f"Saved SGT simulations to {str(save_path)}")

    return siminnov


if __name__ == "__main__":

    seed = 12345
    key = jax.random.key(seed)

    num_sample = 100
    dim = 5

    mean = jnp.repeat(0.0, dim)
    cov = jnp.eye(dim)

    params_z_gaussian = ParamsZGaussian(mean=mean, cov=cov)
    hi = sample_mvar_gaussian(
        key=key, num_sample=num_sample, params_z_gaussian_true=params_z_gaussian
    )

    breakpoint()
