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
from numpy._typing import ArrayLike
import scipy
import matplotlib.pyplot as plt

# HACK:
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod
jax.config.update("jax_debug_nans", True)  # Should disable in full prod


logger = logging.getLogger(__name__)
# logger.basicConfig(filename="./log.log", encoding="utf-8", level=logging.DEBUG)


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
    x: ArrayLike,
    vec_lbda: ArrayLike,
    vec_p0: ArrayLike,
    vec_q0: ArrayLike,
):
    """
    Let X_1 \\sim SGT(\\theta_1), \\ldots, X_d \\sim SGT(\\theta_d) be
    independent. Construct the random vector X = (X_1, \\ldots, X_d).
    Return the density of X.
    """
    _func = vmap(pdf_sgt, in_axes=[0, 0, 0, 0])
    vec_pdf = _func(x, vec_lbda, vec_p0, vec_q0)
    return vec_pdf


def loglik_mvar_indp_sgt(data: Array, vec_lbda: Array, vec_p0: Array, vec_q0: Array):
    """
    (Negative) of the log-likelihood function of a vector of
    independent SGT random variables.
    """
    _func = vmap(pdf_mvar_indp_sgt, in_axes=[0, None, None, None])

    summands = _func(data, vec_lbda, vec_p0, vec_q0)
    loglik_summands = jnp.log(summands)
    loglik = loglik_summands.mean()

    neg_loglik = -1.0 * loglik
    return neg_loglik


def quantile_sgt(
    prob: float,
    lbda: float,
    p0: float,
    q0: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    mean_cent: bool = True,
    var_adj: bool = True,
    use_jax: bool = False,
):
    """
    Univariate SGT quantile
    """
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
    key: KeyArrayLike,
    num_sample: int,
    vec_lbda: ArrayLike,
    vec_p0: ArrayLike,
    vec_q0: ArrayLike,
    mu: float = 0.0,
    sigma: float = 1.0,
    num_cores: int = 1,
) -> Array:
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


def _generate_mvar_sgt_init_conditions(
    key,
    dim: int,
    num_trials: int,
    lbda_a: float = -0.25,
    lbda_b: float = 0.5,
    p0_a: float = 2,
    p0_b: float = 5,
    q0_a: float = 2,
    q0_b: float = 5,
):
    """
    Generate random initial conditions for the MLE estimation
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


def _mvar_sgt_objfun(x, data):
    """
    (Negative) of the log-likelihood of the a vector of
    independent SGT random variables.
    """
    dim = data.shape[1]

    vec_lbda = x[0:dim]
    vec_p0 = x[dim : 2 * dim]
    vec_q0 = x[2 * dim :]

    return loglik_mvar_indp_sgt(
        data=data, vec_lbda=vec_lbda, vec_p0=vec_p0, vec_q0=vec_q0
    )


def mle_mvar_sgt(
    key: KeyArrayLike,
    data: Array,
    num_trials: int,
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
    lst_x0 = _generate_mvar_sgt_init_conditions(key=key, dim=dim, num_trials=num_trials)

    # Multi-start local method
    optres = None
    for ii in range(len(lst_x0)):
        print(f"Iteration num {ii}/{len(lst_x0)}")

        x0 = jnp.array(lst_x0[ii])
        x0 = jnp.ravel(x0)
        try:
            nowres = jscipy.optimize.minimize(
                _mvar_sgt_objfun, x0=x0, method="BFGS", args=(data,), options=options
            )

            if nowres.success and (optres is None or optres.fun > nowres.fun):
                optres = nowres
            else:
                logger.info(f"No solution at iteration {ii}. x0 = {x0}")

        except FloatingPointError:
            logger.warning(f"Iteration {ii} encountered FloatingPointError. x0 = {x0}")
            continue

    if (optres is None) or (optres.success is False):
        logger.error("No solution found!")
    else:
        logger.info("Done! Complete MLE for SGT.")

    return optres


if __name__ == "__main__":
    seed = 1234567
    key = random.key(seed)
    rng = np.random.default_rng(seed)
    num_sample = int(1e3)
    dim = 3
    num_cores = 8

    vec_lbda_true = rng.uniform(-0.25, 0.25, dim)
    vec_p0_true = rng.uniform(2, 10, dim)
    vec_q0_true = rng.uniform(2, 10, dim)

    data = sample_mvar_sgt(
        key=key,
        num_sample=num_sample,
        vec_lbda=vec_lbda_true,
        vec_p0=vec_p0_true,
        vec_q0=vec_q0_true,
        num_cores=num_cores,
    )

    # def objfun(x, data, mu=0.0, sigma=1.0, mean_cent=True, var_adj=True):
    #     lbda = x[0]
    #     p0 = x[1]
    #     q0 = x[2]
    #
    #     neg_loglik = loglik_sgt(
    #         data,
    #         lbda=lbda,
    #         p0=p0,
    #         q0=q0,
    #         mu=mu,
    #         sigma=sigma,
    #         mean_cent=mean_cent,
    #         var_adj=var_adj,
    #     )
    #     return neg_loglik
    #
    #
    # num_params = 3
    # x0 = (1 / 2) * jax.random.uniform(key, shape=(dim * num_params,)) - (1 / 4)
    # x0 = x0.at[1].set(np.abs(x0[1]))
    # x0 = x0.at[2].set(np.abs(x0[2]))

    num_trials = 2

    hi = mle_mvar_sgt(key=key, data=data, num_trials=num_trials)

    breakpoint()
