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

import os
import logging
from pathlib import Path

import itertools
from functools import partial

# import optax
# import jaxopt

import numpy as np
from numpy._typing import ArrayLike
import scipy
import matplotlib.pyplot as plt

# HACK:
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
# jax.config.update("jax_enable_x64", True)  # Should use x64 in full prod
jax.config.update("jax_debug_nans", True)  # Should disable in full prod


logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="sgt.log",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(message)s",
    filemode="w",
)


RUN_TIMEVARYING_SGT_SIMULATIONS = False


def positive_part(x: Array) -> Array:
    """
    Positive part of a scalar x^{+} := \\max\\{ x, 0 \\}
    """
    return jnp.maximum(x, 0)


def negative_part(x: Array) -> Array:
    """
    Negative part of a scalar x^{-} :=
    \\max\\{ -x, 0 \\} = -min\\{ x, 0 \\}
    """
    return -1 * jnp.minimum(x, 0)


def indicator(x):
    """
    Indicator function x \\mapsto \\ind(x \\le 0)
    """
    return x <= 0


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


def loglik_mvar_indp_sgt(
    data: Array, vec_lbda: Array, vec_p0: Array, vec_q0: Array, neg_loglik: bool
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


def quantile_sgt(
    prob: float,
    lbda: float,
    p0: float,
    q0: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    mean_cent: bool = True,
    var_adj: bool = True,
    use_jax: bool = True,
):
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
    key: KeyArrayLike,
    data: Array,
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
    theta: Array, lbda_t_minus_1: Array, z_t_minus_1: Array
) -> Array:
    """
    Time varying dynamics of the \\lambda parameter.
    """
    tanh = jnp.tanh
    arctanh = jnp.arctanh
    log = jnp.log

    _rhs = (
        theta[0]
        + negative_part(theta[1]) * z_t_minus_1 * indicator(z_t_minus_1)
        + positive_part(theta[1]) * z_t_minus_1 * (1 - indicator(z_t_minus_1))
    )

    lbda_t = tanh(_rhs + theta[2] * arctanh(lbda_t_minus_1))
    return lbda_t


def _time_varying_pq_params(
    theta: Array, param_t_minus_1: Array, z_t_minus_1: Array, theta_bar: float = 2.0
) -> Array:
    """
    Time varying dynamics of the p or q parameters.
    """
    abs = jnp.abs
    exp = jnp.exp
    log = jnp.log

    # Define the transition terms on the RHS
    _rhs = (
        log(theta[0])
        + negative_part(theta[1]) * abs(z_t_minus_1) * indicator(z_t_minus_1)
        + positive_part(theta[1]) * abs(z_t_minus_1) * (1 - indicator(z_t_minus_1))
    )

    param_t = theta_bar + exp(_rhs + theta[2] * log(param_t_minus_1 - theta_bar))
    return param_t


def sample_mvar_timevarying_sgt(
    key: KeyArrayLike,
    num_sample: int,
    mat_lbda_tvparams: Array,
    mat_p0_tvparams: Array,
    mat_q0_tvparams: Array,
    vec_lbda_init_t0: Array,
    vec_p0_init_t0: Array,
    vec_q0_init_t0: Array,
    save_path: None | os.PathLike,
) -> tuple[Array, Array, Array, Array]:
    """
    Generate samples of time-varying SGT random
    vectors by inverse transform sampling.

    NOTE: This function does not use JAX.
    """
    dim = jnp.shape(mat_lbda_tvparams)[1]

    # Independent Uniform(0,1) random variables
    unif_data = jax.random.uniform(key=key, shape=(num_sample, dim))

    # Init
    mat_lbda = jnp.empty(shape=(num_sample, dim))
    mat_p0 = jnp.empty(shape=(num_sample, dim))
    mat_q0 = jnp.empty(shape=(num_sample, dim))
    mat_z = jnp.empty(shape=(num_sample, dim))

    mat_lbda = mat_lbda.at[0].set(vec_lbda_init_t0)
    mat_p0 = mat_p0.at[0].set(vec_p0_init_t0)
    mat_q0 = mat_p0.at[0].set(vec_q0_init_t0)
    mat_z = mat_z.at[0].set(vec_z_init_t0)

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
            mat_lbda_tvparams, mat_lbda[tt - 1, :], mat_z[tt - 1, :]
        )

        # Compute p_t
        vec_p0_t = _func_pq(mat_p0_tvparams, mat_p0[tt - 1, :], mat_z[tt - 1, :])

        # Compute q_t
        vec_q0_t = _func_pq(mat_q0_tvparams, mat_q0[tt - 1, :], mat_z[tt - 1, :])

        # Compute z_t
        vec_z_t = quantile_sgt(unif_data[tt, :], vec_lbda_t, vec_p0_t, vec_q0_t)

        # Bookkeeping
        mat_lbda = mat_lbda.at[tt].set(vec_lbda_t)
        mat_p0 = mat_p0.at[tt].set(vec_p0_t)
        mat_q0 = mat_q0.at[tt].set(vec_q0_t)
        mat_z = mat_z.at[tt].set(vec_z_t)

        return mat_lbda, mat_p0, mat_q0, mat_z

    logger.info(f"Begin time-varying SGT simulation")

    carry = (mat_lbda, mat_p0, mat_q0, mat_z)
    for tt in range(1, num_sample):
        carry = _body_fun(tt, carry)

        if tt % 100 == 0:
            logger.info(f"... complete iteration {tt}/{num_sample}")

    data_mat_lbda, data_mat_p0, data_mat_q0, data_mat_z = carry

    logger.info(f"Done time-varying SGT simulation")

    # Save
    if save_path is not None:
        with open(save_path, "wb") as f:
            jnp.savez(
                f,
                num_sample=num_sample,
                dim=dim,
                mat_lbda_tvparams_true=mat_lbda_tvparams_true,
                mat_p0_tvparams_true=mat_p0_tvparams_true,
                mat_q0_tvparams_true=mat_q0_tvparams_true,
                vec_z_init_t0=vec_z_init_t0,
                vec_lbda_init_t0=vec_lbda_init_t0,
                vec_p0_init_t0=vec_p0_init_t0,
                vec_q0_init_t0=vec_q0_init_t0,
                data_mat_lbda=data_mat_lbda,
                data_mat_p0=data_mat_p0,
                data_mat_q0=data_mat_q0,
                data_mat_z=data_mat_z,
            )
        logger.info(f"Saved SGT simulations to {str(save_path)}")

    return data_mat_lbda, data_mat_p0, data_mat_q0, data_mat_z


if __name__ == "__main__":
    if RUN_TIMEVARYING_SGT_SIMULATIONS:
        seed = 1234567
        key = random.key(seed)
        rng = np.random.default_rng(seed)
        num_sample = int(3e3)
        dim = 5
        num_cores = 8
        save_path = Path().resolve() / "data_timevarying_sgt.npz"

        num_lbda_tvparams = 3
        num_p0_tvparams = 3
        num_q0_tvparams = 3

        vec_lbda_true = rng.uniform(-0.25, 0.25, dim)
        vec_p0_true = rng.uniform(2, 10, dim)
        vec_q0_true = rng.uniform(2, 10, dim)

        mat_lbda_tvparams_true = rng.uniform(-0.25, 0.25, (num_lbda_tvparams, dim))
        mat_p0_tvparams_true = rng.uniform(-1, 1, (num_p0_tvparams, dim))
        mat_q0_tvparams_true = rng.uniform(-1, 1, (num_q0_tvparams, dim))
        # mat_lbda_tvparams_true[0, :] = np.abs(mat_lbda_tvparams_true[0, :])
        mat_p0_tvparams_true[0, :] = np.abs(mat_p0_tvparams_true[0, :])
        mat_q0_tvparams_true[0, :] = np.abs(mat_q0_tvparams_true[0, :])

        vec_z_init_t0 = 2 * jax.random.uniform(key, shape=(dim,)) - 1
        vec_z_init_t0 = jnp.repeat(0.0, dim)
        vec_lbda_init_t0 = rng.uniform(-0.25, 0.25, dim)
        vec_p0_init_t0 = rng.uniform(2, 4, dim)
        vec_q0_init_t0 = rng.uniform(2, 4, dim)

        mat_lbda_tvparams = mat_lbda_tvparams_true
        mat_p0_tvparams = mat_p0_tvparams_true
        mat_q0_tvparams = mat_q0_tvparams_true

        data_mat_lbda, data_mat_p0, data_mat_q0, data_mat_z = (
            sample_mvar_timevarying_sgt(
                key=key,
                num_sample=num_sample,
                mat_lbda_tvparams=mat_lbda_tvparams_true,
                mat_p0_tvparams=mat_p0_tvparams_true,
                mat_q0_tvparams=mat_q0_tvparams_true,
                vec_lbda_init_t0=vec_lbda_init_t0,
                vec_p0_init_t0=vec_p0_init_t0,
                vec_q0_init_t0=vec_q0_init_t0,
                save_path=save_path,
            )
        )

    # Load simulations
    save_path = Path().resolve() / "data_timevarying_sgt.npz"
    with open(save_path, "rb") as f:
        npzfile = jnp.load(f)
