from enum import verify
from threading import excepthook
import jax
from jax._src.random import KeyArrayLike
import jax.numpy as jnp
import jax.scipy as jscipy
import jax.scipy.optimize
from jax import grad, jit, vmap
from jax import random
import jax.test_util
import jaxopt
import jaxtyping as jpt

import logging
import os
import pathlib
import pickle

import numpy as np

import dataclasses
from dataclasses import dataclass

import dcc
import sgt
from sgt import ParamsZSgt

logger = logging.getLogger(__name__)
logging.basicConfig(
    datefmt="%Y-%m-%d %I:%M:%S %p",
    level=logging.INFO,
    format="%(levelname)s | %(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/estimate_dcc_sgt_garch.log", mode="w"),
        logging.StreamHandler(),
    ],
)


seed = 987654321
key = random.key(seed)
rng = np.random.default_rng(seed)
num_cores = 8

# Read in simulated results
data_simreturns_savepath = (
    pathlib.Path().resolve() / "simulated_data/data_simreturns_timevarying_sgt.pkl"
)
with open(data_simreturns_savepath, "rb") as f:
    simreturns = pickle.load(f)

num_sample = simreturns.num_sample
dim = simreturns.dim

# Initial guess of the parameters of the time-varying SGT process
mat_lbda_tvparams = rng.uniform(-0.25, 0.25, (sgt.NUM_LBDA_TVPARAMS, dim))
mat_lbda_tvparams[0, :] = np.abs(mat_lbda_tvparams[0, :])
mat_p0_tvparams = rng.uniform(-0.25, 0.25, (sgt.NUM_P0_TVPARAMS, dim))
mat_p0_tvparams[0, :] = np.abs(mat_p0_tvparams[0, :])
mat_q0_tvparams = rng.uniform(-0.25, 0.25, (sgt.NUM_Q0_TVPARAMS, dim))
mat_q0_tvparams[0, :] = np.abs(mat_q0_tvparams[0, :])
params_z_sgt_init_guess = sgt.ParamsZSgt(
    mat_lbda_tvparams=jnp.array(mat_lbda_tvparams),
    mat_p0_tvparams=jnp.array(mat_p0_tvparams),
    mat_q0_tvparams=jnp.array(mat_q0_tvparams),
)

# Initial guess for parameters for the mean returns vector
params_mean_init_guess = dcc.ParamsMean(vec_mu=jnp.array(rng.uniform(0, 1, dim) / 50))

# Initial guess for params for DCC -- univariate vols
params_uvar_vol_init_guess = dcc.ParamsUVarVol(
    vec_omega=jnp.array(rng.uniform(0, 1, dim) / 2),
    vec_beta=jnp.array(rng.uniform(0, 1, dim) / 3),
    vec_alpha=jnp.array(rng.uniform(0, 1, dim) / 10),
    vec_psi=jnp.array(rng.uniform(0, 1, dim) / 5),
)
# Initial guess for params for DCC -- multivariate Q
params_mvar_cor_init_guess = dcc.ParamsMVarCor(
    vec_delta=jnp.array([0.054, 0.230]),
    mat_Qbar=dcc.generate_random_cov_mat(key=key, dim=dim) / 5,
)

# Package all the initial guess DCC params together
params_dcc_init_guess = dcc.ParamsDcc(
    uvar_vol=params_uvar_vol_init_guess,
    mvar_cor=params_mvar_cor_init_guess,
)

params_dcc_sgt_garch_init_guess = dcc.ParamsDccSgtGarch(
    sgt=params_z_sgt_init_guess,
    mean=params_mean_init_guess,
    dcc=params_dcc_init_guess,
)


# Initial t = 0 conditions for the DCC Q_t process
subkeys = random.split(key, 2)
mat_Sigma_init_t0_guess = dcc.generate_random_cov_mat(key=subkeys[0], dim=dim)
mat_Q_init_t0_guess = dcc.generate_random_cov_mat(key=subkeys[1], dim=dim)
inittimecond_dcc_guess = dcc.InitTimeConditionDcc(
    mat_Sigma_init_t0=mat_Sigma_init_t0_guess, mat_Q_init_t0=mat_Q_init_t0_guess
)

# Initital t = 0 conditions for the SGT time-varying
# parameters process
inittimecond_z_sgt_guess = sgt.InitTimeConditionZSgt(
    vec_z_init_t0=jnp.repeat(0.0, dim),
    vec_lbda_init_t0=jnp.array(rng.uniform(-0.25, 0.25, dim)),
    vec_p0_init_t0=jnp.array(rng.uniform(2, 4, dim)),
    vec_q0_init_t0=jnp.array(rng.uniform(2, 4, dim)),
)

inittimecond_dcc_sgt_garch_guess = dcc.InitTimeConditionDccSgtGarch(
    sgt=inittimecond_z_sgt_guess, dcc=inittimecond_dcc_guess
)

mat_returns = simreturns.data_mat_returns


def params_to_arr(
    params_dataclass: (
        sgt.ParamsZSgt | dcc.ParamsMean | dcc.ParamsUVarVol | dcc.ParamsMVarCor
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
    return arr


# hi = dcc._make_params_from_arr_z_sgt(x=jnp.repeat(0.25, (3 + 3 + 3) * dim), dim=dim)


# hi = dcc._objfun_dcc_loglik(
#     x=jnp.repeat(0.25, (3 + 3 + 3) * dim),
#     make_params_from_arr=dcc._make_params_from_arr_z_sgt,
#     params_dcc_sgt_garch=params_dcc_sgt_garch_init_guess,
#     inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch_guess,
#     mat_returns=mat_returns,
# )

neg_loglik_optval, params_dcc_sgt_garch_opt = dcc.dcc_sgt_garch_optimization(
    mat_returns=mat_returns,
    params_dcc_sgt_garch=params_dcc_sgt_garch_init_guess,
    inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch_guess,
    verbose=False,
)

breakpoint()


def bunny(
    x,
    params_z_sgt: dcc.ParamsZSgt,
    mat_returns,
    params_dcc_sgt_garch: dcc.ParamsDccSgtGarch,
):
    params_dcc_sgt_garch.sgt = params_z_sgt

    val = dcc.dcc_sgt_loglik(
        params_dcc_sgt_garch=params_dcc_sgt_garch_init_guess,
        mat_returns=mat_returns,
        inittimecond_dcc_sgt_garch=inittimecond_dcc_sgt_garch_guess,
    )
    return val


solver = jaxopt.LBFGS
verbose = True
solver_obj = solver(bunny, verbose=verbose)
res = solver_obj.run(
    params_z_sgt_init_guess,
    mat_returns=mat_returns,
    params_dcc_sgt_garch=params_dcc_sgt_garch_init_guess,
)

breakpoint()


# neg_loglik_optval, dict_params = dcc.dcc_sgt_garch_optimization(
#     mat_returns=mat_returns,
#     params_init_guess=params_dcc_sgt_garch_init_guess,
#     # dict_params_init_guess=dict_params_init_guess,
#     # dict_init_t0_conditions=dict_init_t0_conditions,
# )
